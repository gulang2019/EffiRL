#!/usr/bin/env python3
"""Train a LoRA classifier to predict average generation-length bins."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LengthExample:
    checkpoint_step: int
    prompt_index: int
    example_key: str
    prompt_messages: list[dict[str, str]]
    average_generation_length: float
    label: int


class LengthDataset(Dataset):
    def __init__(self, examples: list[LengthExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LengthExample:
        return self.examples[index]


class LengthClassifier(nn.Module):
    def __init__(self, backbone, num_labels: int):
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1]
        last_token_index = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), last_token_index]
        pooled = self.dropout(pooled)
        return self.classifier(pooled.float())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=Path("outputs/profiling_gsm8k_efficiency_2026-03-27_v32_len1024/ground_truth_profile.csv"),
    )
    parser.add_argument(
        "--data-files",
        type=Path,
        nargs="+",
        default=[Path("data/verl/gsm8k/test.parquet"), Path("data/verl/math/test.parquet")],
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/EffiRL/qwen2.5-1.5b-grpo-math-single-gpu-vllm"),
    )
    parser.add_argument("--init-checkpoint-step", type=int, default=200)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=4)
    parser.add_argument("--train-prompt-fraction", type=float, default=0.7)
    parser.add_argument("--val-prompt-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/profiling_gsm8k_efficiency_2026-03-27_v32_len1024/analysis/avg_length_lora_classifier"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt_messages(data_files: list[Path]) -> dict[str, list[dict[str, str]]]:
    mapping: dict[str, list[dict[str, str]]] = {}
    for path in data_files:
        table = pq.read_table(path)
        for row in table.to_pylist():
            extra_info = row.get("extra_info") or {}
            key = f"{row['data_source']}::{extra_info.get('index', -1)}"
            mapping[key] = row["prompt"]
    return mapping


def assign_balanced_bins(values: list[float], num_bins: int) -> tuple[list[int], list[dict[str, float]]]:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    labels = [0 for _ in values]
    summaries: list[dict[str, float]] = []
    n = len(values)
    for bin_idx in range(num_bins):
        start = math.floor(bin_idx * n / num_bins)
        end = math.floor((bin_idx + 1) * n / num_bins)
        bin_indices = order[start:end]
        for idx in bin_indices:
            labels[idx] = bin_idx
        bin_values = [values[idx] for idx in bin_indices]
        summaries.append(
            {
                "label": bin_idx,
                "count": len(bin_indices),
                "min_avg_length": float(min(bin_values)) if bin_values else 0.0,
                "max_avg_length": float(max(bin_values)) if bin_values else 0.0,
                "mean_avg_length": float(sum(bin_values) / len(bin_values)) if bin_values else 0.0,
            }
        )
    return labels, summaries


def load_examples(profile_csv: Path, prompt_messages_by_key: dict[str, list[dict[str, str]]], num_bins: int) -> tuple[list[LengthExample], list[dict[str, float]]]:
    raw_rows: list[dict[str, Any]] = []
    avg_lengths: list[float] = []
    with profile_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            completion_lengths = json.loads(row["completion_lengths"])
            avg_length = float(sum(completion_lengths) / len(completion_lengths))
            raw_rows.append(row)
            avg_lengths.append(avg_length)
    labels, bin_summaries = assign_balanced_bins(avg_lengths, num_bins)

    examples: list[LengthExample] = []
    for row, avg_length, label in zip(raw_rows, avg_lengths, labels, strict=True):
        key = row["example_key"]
        prompt_messages = prompt_messages_by_key.get(key)
        if prompt_messages is None:
            raise KeyError(f"Missing prompt messages for example_key={key}")
        examples.append(
            LengthExample(
                checkpoint_step=int(row["checkpoint_step"]),
                prompt_index=int(row["prompt_index"]),
                example_key=key,
                prompt_messages=prompt_messages,
                average_generation_length=avg_length,
                label=label,
            )
        )
    return examples, bin_summaries


def build_splits(examples: list[LengthExample], train_fraction: float, val_fraction: float, seed: int) -> dict[str, list[LengthExample]]:
    unique_prompts = sorted({example.prompt_index for example in examples})
    rng = random.Random(seed)
    rng.shuffle(unique_prompts)

    total_prompts = len(unique_prompts)
    train_count = max(1, int(round(total_prompts * train_fraction)))
    val_count = max(1, int(round(total_prompts * val_fraction)))
    if train_count + val_count >= total_prompts:
        val_count = max(1, total_prompts - train_count - 1)
    test_count = total_prompts - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1

    train_prompts = set(unique_prompts[:train_count])
    val_prompts = set(unique_prompts[train_count : train_count + val_count])
    test_prompts = set(unique_prompts[train_count + val_count :])

    return {
        "train": [example for example in examples if example.prompt_index in train_prompts],
        "val": [example for example in examples if example.prompt_index in val_prompts],
        "test": [example for example in examples if example.prompt_index in test_prompts],
    }


def make_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_backbone(base_model: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
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
    model.to(device)
    return model


def load_checkpoint_into_backbone(model, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load checkpoint cleanly: {checkpoint_path}\nmissing={missing[:10]}\nunexpected={unexpected[:10]}"
        )


def render_classifier_prompt(tokenizer, example: LengthExample) -> str:
    messages = [
        {"role": "system", "content": f"Checkpoint step: {example.checkpoint_step}."},
        *example.prompt_messages,
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_collate_fn(tokenizer, max_prompt_tokens: int):
    def collate(examples: list[LengthExample]) -> dict[str, Any]:
        texts = [render_classifier_prompt(tokenizer, example) for example in examples]
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        batch["labels"] = torch.tensor([example.label for example in examples], dtype=torch.long)
        batch["average_generation_length"] = torch.tensor(
            [example.average_generation_length for example in examples],
            dtype=torch.float32,
        )
        batch["checkpoint_step"] = torch.tensor([example.checkpoint_step for example in examples], dtype=torch.long)
        batch["prompt_index"] = torch.tensor([example.prompt_index for example in examples], dtype=torch.long)
        batch["example_key"] = [example.example_key for example in examples]
        return batch

    return collate


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            pred_labels = logits.argmax(dim=-1)
            total_correct += int((pred_labels == labels).sum().item())
            total_examples += labels.size(0)
            probabilities = torch.softmax(logits, dim=-1).cpu()
            for idx in range(labels.size(0)):
                predictions.append(
                    {
                        "example_key": batch["example_key"][idx],
                        "checkpoint_step": int(batch["checkpoint_step"][idx]),
                        "prompt_index": int(batch["prompt_index"][idx]),
                        "true_label": int(labels[idx].cpu().item()),
                        "predicted_label": int(pred_labels[idx].cpu().item()),
                        "average_generation_length": float(batch["average_generation_length"][idx]),
                        "confidence": float(probabilities[idx, pred_labels[idx].cpu().item()].item()),
                    }
                )
    metrics = {
        "loss": total_loss / total_examples if total_examples else 0.0,
        "accuracy": total_correct / total_examples if total_examples else 0.0,
        "count": total_examples,
    }
    return metrics, predictions


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    prompt_messages_by_key = load_prompt_messages(args.data_files)
    examples, bin_summaries = load_examples(args.profile_csv, prompt_messages_by_key, args.num_bins)
    splits = build_splits(examples, args.train_prompt_fraction, args.val_prompt_fraction, args.seed)

    tokenizer = make_tokenizer(args.base_model)
    device = torch.device(args.device)
    backbone = build_backbone(args.base_model, device)
    checkpoint_path = args.checkpoint_dir / f"global_step_{args.init_checkpoint_step}" / "actor" / "model_world_size_1_rank_0.pt"
    load_checkpoint_into_backbone(backbone, checkpoint_path)

    model = LengthClassifier(backbone=backbone, num_labels=args.num_bins).to(device)

    collate_fn = make_collate_fn(tokenizer, args.max_prompt_tokens)
    train_loader = DataLoader(LengthDataset(splits["train"]), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(LengthDataset(splits["val"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(LengthDataset(splits["test"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    training_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_examples = 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            epoch_loss += float(loss.item()) * labels.size(0)
            epoch_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            epoch_examples += labels.size(0)

        train_metrics = {
            "loss": epoch_loss / epoch_examples if epoch_examples else 0.0,
            "accuracy": epoch_correct / epoch_examples if epoch_examples else 0.0,
        }
        val_metrics, _ = evaluate(model, val_loader, device)
        training_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)

    train_metrics, train_predictions = evaluate(model, train_loader, device)
    val_metrics, val_predictions = evaluate(model, val_loader, device)
    test_metrics, test_predictions = evaluate(model, test_loader, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "adapter"
    model.backbone.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "dropout_p": model.dropout.p,
            "num_labels": args.num_bins,
            "bin_summaries": bin_summaries,
            "init_checkpoint_step": args.init_checkpoint_step,
        },
        args.output_dir / "classifier_head.pt",
    )

    write_csv(training_rows, args.output_dir / "training_log.csv")
    write_csv(train_predictions, args.output_dir / "train_predictions.csv")
    write_csv(val_predictions, args.output_dir / "val_predictions.csv")
    write_csv(test_predictions, args.output_dir / "test_predictions.csv")

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "num_bins": args.num_bins,
        "bin_summaries": bin_summaries,
        "train_examples": len(splits["train"]),
        "val_examples": len(splits["val"]),
        "test_examples": len(splits["test"]),
        "train_prompt_count": len({example.prompt_index for example in splits["train"]}),
        "val_prompt_count": len({example.prompt_index for example in splits["val"]}),
        "test_prompt_count": len({example.prompt_index for example in splits["test"]}),
        "init_checkpoint_step": args.init_checkpoint_step,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    split_manifest = {
        name: sorted({example.prompt_index for example in subset})
        for name, subset in splits.items()
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    print(f"Wrote adapter to {adapter_dir}")
    print(f"Wrote metrics to {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
