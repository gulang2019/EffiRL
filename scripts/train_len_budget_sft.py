#!/usr/bin/env python3
"""Train a LoRA SFT adapter for len-budget prefix prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


FORECAST_RE = re.compile(r"<forecast\s+len_budget=(lb_\d+)>")
IGNORE_INDEX = -100


@dataclass
class LenBudgetExample:
    prompt_messages: list[dict[str, str]]
    assistant_content: str
    example_key: str
    prompt_index: int
    len_budget_bin: str
    declared_budget_tokens: int
    meta: dict[str, Any]


class LenBudgetDataset(Dataset):
    def __init__(self, examples: list[LenBudgetExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LenBudgetExample:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("outputs/len_budget_sft_math_ckpt200_partial_snapshot/train.jsonl"),
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("outputs/len_budget_sft_math_ckpt200_partial_snapshot/val.jsonl"),
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=Path("outputs/len_budget_sft_math_ckpt200_partial_snapshot/test.jsonl"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/EffiRL/qwen2.5-1.5b-grpo-math-single-gpu-vllm"),
    )
    parser.add_argument("--init-checkpoint-step", type=int, default=200)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-completion-tokens", type=int, default=1152)
    parser.add_argument("--append-eos", action="store_true", default=True)
    parser.add_argument("--disable-append-eos", action="store_false", dest="append_eos")
    parser.add_argument("--prefix-weight", type=float, default=5.0)
    parser.add_argument("--answer-weight", type=float, default=1.0)
    parser.add_argument(
        "--prefix-only-warmup-ratio",
        type=float,
        default=0.25,
        help="Fraction of optimizer steps that use zero answer-token loss.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--disable-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--eval-forecast-max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-generation-batch-size", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-val-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/len_budget_sft_math_ckpt200_partial_snapshot/adapter_run"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_examples(path: Path, limit: int) -> list[LenBudgetExample]:
    examples: list[LenBudgetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            messages = row["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Invalid messages field in {path}: {row}")
            assistant = messages[-1]
            if assistant.get("role") != "assistant":
                raise ValueError(f"Expected assistant as last message in {path}: {assistant}")
            meta = dict(row.get("meta") or {})
            examples.append(
                LenBudgetExample(
                    prompt_messages=messages[:-1],
                    assistant_content=str(assistant["content"]),
                    example_key=str(meta["example_key"]),
                    prompt_index=int(meta["prompt_index"]),
                    len_budget_bin=str(meta["len_budget_bin"]),
                    declared_budget_tokens=int(meta["declared_budget_tokens"]),
                    meta=meta,
                )
            )
            if limit > 0 and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def make_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(base_model: str, device: torch.device, torch_dtype: torch.dtype, gradient_checkpointing: bool):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch_dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device)
    return model


def load_checkpoint_into_model(model, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load checkpoint cleanly: {checkpoint_path}\nmissing={missing[:10]}\nunexpected={unexpected[:10]}"
        )


def split_assistant_content(assistant_content: str) -> tuple[str, str]:
    if "\n" in assistant_content:
        first_line, remainder = assistant_content.split("\n", 1)
        return first_line + "\n", remainder
    return assistant_content, ""


def render_prompt(tokenizer, prompt_messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def make_collate_fn(tokenizer, max_prompt_tokens: int, max_completion_tokens: int, append_eos: bool):
    eos_token_id = tokenizer.eos_token_id
    if append_eos and eos_token_id is None:
        raise ValueError("append_eos=True requires tokenizer.eos_token_id to be set.")

    def collate(examples: list[LenBudgetExample]) -> dict[str, Any]:
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_prefix_mask: list[list[float]] = []
        batch_answer_mask: list[list[float]] = []
        batch_attention_mask: list[list[int]] = []
        prompt_texts: list[str] = []
        target_bins: list[str] = []
        forecast_texts: list[str] = []
        metas: list[dict[str, Any]] = []

        for example in examples:
            prompt_text = render_prompt(tokenizer, example.prompt_messages)
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_tokens,
            ).input_ids

            forecast_text, answer_text = split_assistant_content(example.assistant_content)
            forecast_ids = tokenizer(forecast_text, add_special_tokens=False).input_ids
            eos_extra = 1 if append_eos else 0
            available_answer_tokens = max_completion_tokens - len(forecast_ids) - eos_extra
            if available_answer_tokens < 0:
                raise ValueError(
                    f"Forecast prefix is longer than max_completion_tokens={max_completion_tokens} for {example.example_key}."
                )
            answer_ids = tokenizer(
                answer_text,
                add_special_tokens=False,
                truncation=True,
                max_length=available_answer_tokens,
            ).input_ids

            completion_ids = list(forecast_ids) + list(answer_ids)
            if append_eos:
                completion_ids.append(eos_token_id)

            input_ids = list(prompt_ids) + completion_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
            prefix_mask = [0.0] * len(prompt_ids) + [1.0] * len(forecast_ids) + [0.0] * (len(completion_ids) - len(forecast_ids))
            answer_mask = [0.0] * len(prompt_ids) + [0.0] * len(forecast_ids) + [1.0] * (len(completion_ids) - len(forecast_ids))
            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_prefix_mask.append(prefix_mask)
            batch_answer_mask.append(answer_mask)
            batch_attention_mask.append(attention_mask)
            prompt_texts.append(prompt_text)
            target_bins.append(example.len_budget_bin)
            forecast_texts.append(forecast_text)
            metas.append(example.meta)

        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = int(tokenizer.pad_token_id)
        for idx in range(len(batch_input_ids)):
            pad_len = max_len - len(batch_input_ids[idx])
            if pad_len <= 0:
                continue
            batch_input_ids[idx] = batch_input_ids[idx] + [pad_id] * pad_len
            batch_labels[idx] = batch_labels[idx] + [IGNORE_INDEX] * pad_len
            batch_prefix_mask[idx] = batch_prefix_mask[idx] + [0.0] * pad_len
            batch_answer_mask[idx] = batch_answer_mask[idx] + [0.0] * pad_len
            batch_attention_mask[idx] = batch_attention_mask[idx] + [0] * pad_len

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "prefix_mask": torch.tensor(batch_prefix_mask, dtype=torch.float32),
            "answer_mask": torch.tensor(batch_answer_mask, dtype=torch.float32),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "prompt_text": prompt_texts,
            "target_bin": target_bins,
            "forecast_text": forecast_texts,
            "meta": metas,
        }

    return collate


def weighted_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prefix_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    prefix_weight: float,
    answer_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    shift_prefix_mask = prefix_mask[:, 1:]
    shift_answer_mask = answer_mask[:, 1:]

    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view_as(shift_labels)

    token_weights = shift_prefix_mask * float(prefix_weight) + shift_answer_mask * float(answer_weight)
    valid_mask = shift_labels.ne(IGNORE_INDEX)
    token_weights = token_weights * valid_mask.float()

    weight_sum = token_weights.sum().clamp(min=1e-8)
    loss = (token_loss * token_weights).sum() / weight_sum

    with torch.no_grad():
        predictions = shift_logits.argmax(dim=-1)
        correct = predictions.eq(shift_labels) & valid_mask
        prefix_valid = shift_prefix_mask > 0
        answer_valid = shift_answer_mask > 0
        metrics = {
            "weighted_loss": float(loss.item()),
            "prefix_token_count": float(prefix_valid.sum().item()),
            "answer_token_count": float(answer_valid.sum().item()),
            "prefix_token_accuracy": float(correct[prefix_valid].float().mean().item()) if prefix_valid.any() else 0.0,
            "answer_token_accuracy": float(correct[answer_valid].float().mean().item()) if answer_valid.any() else 0.0,
        }
    return loss, metrics


def evaluate_loss(
    model,
    data_loader: DataLoader,
    device: torch.device,
    prefix_weight: float,
    answer_weight: float,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    prefix_token_weighted_accuracy = 0.0
    answer_token_weighted_accuracy = 0.0
    prefix_token_total = 0.0
    answer_token_total = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                use_cache=False,
            )
            _, batch_metrics = weighted_language_model_loss(
                logits=outputs.logits,
                labels=batch["labels"].to(device),
                prefix_mask=batch["prefix_mask"].to(device),
                answer_mask=batch["answer_mask"].to(device),
                prefix_weight=prefix_weight,
                answer_weight=answer_weight,
            )
            loss_sum += batch_metrics["weighted_loss"]
            prefix_token_total += batch_metrics["prefix_token_count"]
            answer_token_total += batch_metrics["answer_token_count"]
            prefix_token_weighted_accuracy += batch_metrics["prefix_token_accuracy"] * batch_metrics["prefix_token_count"]
            answer_token_weighted_accuracy += batch_metrics["answer_token_accuracy"] * batch_metrics["answer_token_count"]
            batch_count += 1

    return {
        "loss": loss_sum / batch_count if batch_count else 0.0,
        "prefix_token_accuracy": prefix_token_weighted_accuracy / prefix_token_total if prefix_token_total else 0.0,
        "answer_token_accuracy": answer_token_weighted_accuracy / answer_token_total if answer_token_total else 0.0,
        "prefix_token_count": prefix_token_total,
        "answer_token_count": answer_token_total,
    }


def decode_generated_suffix(tokenizer, suffix_ids: list[int], eos_token_id: int | None) -> str:
    decoded_ids: list[int] = []
    for token_id in suffix_ids:
        if eos_token_id is not None and token_id == eos_token_id:
            break
        decoded_ids.append(token_id)
    return tokenizer.decode(decoded_ids, skip_special_tokens=True)


def evaluate_forecast_generation(
    model,
    tokenizer,
    examples: list[LenBudgetExample],
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not examples:
        return {
            "count": 0,
            "parse_rate": 0.0,
            "exact_match_rate": 0.0,
        }, []

    tokenizer.padding_side = "left"
    eos_token_id = tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []
    parsed = 0
    exact = 0

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(examples), batch_size):
            subset = examples[batch_start : batch_start + batch_size]
            prompt_texts = [
                tokenizer.apply_chat_template(example.prompt_messages, tokenize=False, add_generation_prompt=True)
                for example in subset
            ]
            tokenized = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
                add_special_tokens=False,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            padded_prompt_len = input_ids.shape[1]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
            sequences = outputs.detach().cpu().tolist()
            for example, sequence in zip(subset, sequences, strict=True):
                decoded_text = decode_generated_suffix(tokenizer, sequence[padded_prompt_len:], eos_token_id)
                match = FORECAST_RE.search(decoded_text)
                predicted_bin = match.group(1) if match else None
                if predicted_bin is not None:
                    parsed += 1
                if predicted_bin == example.len_budget_bin:
                    exact += 1
                rows.append(
                    {
                        "example_key": example.example_key,
                        "prompt_index": example.prompt_index,
                        "target_bin": example.len_budget_bin,
                        "predicted_bin": predicted_bin,
                        "parse_success": int(predicted_bin is not None),
                        "exact_match": int(predicted_bin == example.len_budget_bin),
                        "generated_prefix_text": decoded_text,
                    }
                )

    tokenizer.padding_side = "right"
    total = len(examples)
    metrics = {
        "count": total,
        "parse_rate": parsed / total,
        "exact_match_rate": exact / total,
    }
    return metrics, rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = args.output_dir / "training_log.csv"
    progress_path = args.output_dir / "progress.json"

    train_examples = read_examples(args.train_jsonl, args.max_train_examples)
    val_examples = read_examples(args.val_jsonl, args.max_val_examples)
    test_examples = read_examples(args.test_jsonl, args.max_test_examples)

    tokenizer = make_tokenizer(args.base_model)
    device = torch.device(args.device)
    torch_dtype = resolve_torch_dtype(args.dtype, device)
    model = build_model(args.base_model, device, torch_dtype, args.gradient_checkpointing)
    checkpoint_path = args.checkpoint_dir / f"global_step_{args.init_checkpoint_step}" / "actor" / "model_world_size_1_rank_0.pt"
    load_checkpoint_into_model(model, checkpoint_path)

    collate_fn = make_collate_fn(
        tokenizer=tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
        append_eos=args.append_eos,
    )
    train_loader = DataLoader(LenBudgetDataset(train_examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(LenBudgetDataset(val_examples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(LenBudgetDataset(test_examples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_train_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(round(total_train_steps * args.prefix_only_warmup_ratio))
    training_rows: list[dict[str, Any]] = []
    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    global_step = 0
    run_started_at = time.time()

    write_json(
        {
            "state": "running",
            "phase": "starting",
            "pid": os.getpid(),
            "started_at": run_started_at,
            "epoch": 0,
            "global_step": 0,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
            "output_dir": str(args.output_dir),
        },
        progress_path,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_prefix_acc_sum = 0.0
        train_answer_acc_sum = 0.0
        prefix_token_total = 0.0
        answer_token_total = 0.0
        batch_count = 0
        epoch_start = time.time()

        for batch in train_loader:
            global_step += 1
            effective_answer_weight = 0.0 if global_step <= warmup_steps else args.answer_weight

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                use_cache=False,
            )
            loss, batch_metrics = weighted_language_model_loss(
                logits=outputs.logits,
                labels=batch["labels"].to(device),
                prefix_mask=batch["prefix_mask"].to(device),
                answer_mask=batch["answer_mask"].to(device),
                prefix_weight=args.prefix_weight,
                answer_weight=effective_answer_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            train_loss_sum += batch_metrics["weighted_loss"]
            train_prefix_acc_sum += batch_metrics["prefix_token_accuracy"] * batch_metrics["prefix_token_count"]
            train_answer_acc_sum += batch_metrics["answer_token_accuracy"] * batch_metrics["answer_token_count"]
            prefix_token_total += batch_metrics["prefix_token_count"]
            answer_token_total += batch_metrics["answer_token_count"]
            batch_count += 1

            if args.log_interval > 0 and (batch_count % args.log_interval == 0 or batch_count == len(train_loader)):
                running_train_row = {
                    "state": "running",
                    "phase": "train",
                    "pid": os.getpid(),
                    "started_at": run_started_at,
                    "elapsed_seconds": time.time() - run_started_at,
                    "epoch": epoch,
                    "epoch_batches_completed": batch_count,
                    "epoch_batches_total": len(train_loader),
                    "global_step": global_step,
                    "warmup_steps": warmup_steps,
                    "running_train_loss": train_loss_sum / batch_count if batch_count else 0.0,
                    "running_train_prefix_token_accuracy": (
                        train_prefix_acc_sum / prefix_token_total if prefix_token_total else 0.0
                    ),
                    "running_train_answer_token_accuracy": (
                        train_answer_acc_sum / answer_token_total if answer_token_total else 0.0
                    ),
                    "latest_val_loss": training_rows[-1]["val_loss"] if training_rows else None,
                    "latest_val_prefix_token_accuracy": (
                        training_rows[-1]["val_prefix_token_accuracy"] if training_rows else None
                    ),
                    "latest_val_answer_token_accuracy": (
                        training_rows[-1]["val_answer_token_accuracy"] if training_rows else None
                    ),
                }
                write_json(running_train_row, progress_path)
                print(
                    f"[train epoch {epoch} step {batch_count}/{len(train_loader)}] "
                    f"loss={running_train_row['running_train_loss']:.4f} "
                    f"prefix_acc={running_train_row['running_train_prefix_token_accuracy']:.3f} "
                    f"answer_acc={running_train_row['running_train_answer_token_accuracy']:.3f}",
                    flush=True,
                )

        val_metrics = evaluate_loss(
            model=model,
            data_loader=val_loader,
            device=device,
            prefix_weight=args.prefix_weight,
            answer_weight=args.answer_weight,
        )
        epoch_row = {
            "epoch": epoch,
            "global_step": global_step,
            "warmup_steps": warmup_steps,
            "train_loss": train_loss_sum / batch_count if batch_count else 0.0,
            "train_prefix_token_accuracy": train_prefix_acc_sum / prefix_token_total if prefix_token_total else 0.0,
            "train_answer_token_accuracy": train_answer_acc_sum / answer_token_total if answer_token_total else 0.0,
            "val_loss": val_metrics["loss"],
            "val_prefix_token_accuracy": val_metrics["prefix_token_accuracy"],
            "val_answer_token_accuracy": val_metrics["answer_token_accuracy"],
        }
        training_rows.append(epoch_row)
        append_csv_row(epoch_row, training_log_path)
        print(
            f"[epoch {epoch}] train_loss={epoch_row['train_loss']:.4f} "
            f"val_loss={epoch_row['val_loss']:.4f} "
            f"val_prefix_acc={epoch_row['val_prefix_token_accuracy']:.3f} "
            f"val_answer_acc={epoch_row['val_answer_token_accuracy']:.3f}",
            flush=True,
        )
        write_json(
            {
                "state": "running",
                "phase": "epoch_eval",
                "pid": os.getpid(),
                "started_at": run_started_at,
                "elapsed_seconds": time.time() - run_started_at,
                "epoch": epoch,
                "epoch_seconds": time.time() - epoch_start,
                "global_step": global_step,
                **epoch_row,
            },
            progress_path,
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in get_peft_model_state_dict(model).items()
            }

    assert best_state is not None
    set_peft_model_state_dict(model, best_state)

    train_metrics = evaluate_loss(model, train_loader, device, args.prefix_weight, args.answer_weight)
    val_metrics = evaluate_loss(model, val_loader, device, args.prefix_weight, args.answer_weight)
    test_metrics = evaluate_loss(model, test_loader, device, args.prefix_weight, args.answer_weight)
    val_generation_metrics, val_generation_rows = evaluate_forecast_generation(
        model=model,
        tokenizer=tokenizer,
        examples=val_examples,
        batch_size=args.eval_generation_batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.eval_forecast_max_new_tokens,
        device=device,
    )
    test_generation_metrics, test_generation_rows = evaluate_forecast_generation(
        model=model,
        tokenizer=tokenizer,
        examples=test_examples,
        batch_size=args.eval_generation_batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.eval_forecast_max_new_tokens,
        device=device,
    )

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    write_csv(val_generation_rows, args.output_dir / "val_forecast_predictions.csv")
    write_csv(test_generation_rows, args.output_dir / "test_forecast_predictions.csv")

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "val_generation": val_generation_metrics,
        "test_generation": test_generation_metrics,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "train_prompt_count": len({example.prompt_index for example in train_examples}),
        "val_prompt_count": len({example.prompt_index for example in val_examples}),
        "test_prompt_count": len({example.prompt_index for example in test_examples}),
        "prefix_weight": args.prefix_weight,
        "answer_weight": args.answer_weight,
        "prefix_only_warmup_ratio": args.prefix_only_warmup_ratio,
        "warmup_steps": warmup_steps,
        "init_checkpoint_step": args.init_checkpoint_step,
        "dtype": str(torch_dtype),
        "device": str(device),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_json(
        {
            "state": "completed",
            "phase": "done",
            "pid": os.getpid(),
            "started_at": run_started_at,
            "elapsed_seconds": time.time() - run_started_at,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "val_generation_exact_match_rate": val_generation_metrics["exact_match_rate"],
            "test_generation_exact_match_rate": test_generation_metrics["exact_match_rate"],
            "metrics_path": str(args.output_dir / "metrics.json"),
            "adapter_dir": str(adapter_dir),
        },
        progress_path,
    )

    split_manifest = {
        "train_example_keys": sorted({example.example_key for example in train_examples}),
        "val_example_keys": sorted({example.example_key for example in val_examples}),
        "test_example_keys": sorted({example.example_key for example in test_examples}),
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    print(f"Wrote adapter to {adapter_dir}")
    print(f"Wrote metrics to {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
