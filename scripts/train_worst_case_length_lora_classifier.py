#!/usr/bin/env python3
"""Train a LoRA classifier for expected worst-case generation length quartiles."""

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
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class WorstCaseExample:
    prompt_index: int
    example_key: str
    prompt_text: str
    prompt_tokens: int
    expected_worst_case_length: float
    label: int = -1


class WorstCaseDataset(Dataset):
    def __init__(self, examples: list[WorstCaseExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> WorstCaseExample:
        return self.examples[index]


class WorstCaseClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 1)

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
        return self.classifier(pooled.float()).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-csv",
        type=Path,
        default=Path("outputs/worst_case_length_pool_gsm8k_ckpt200/worst_case_length_profile.csv"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/EffiRL/qwen2.5-1.5b-grpo-math-single-gpu-vllm"),
    )
    parser.add_argument("--init-checkpoint-step", type=int, default=200)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--train-prompt-fraction", type=float, default=0.7)
    parser.add_argument("--val-prompt-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/worst_case_length_pool_gsm8k_ckpt200/quartile_classifier"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(target_csv: Path) -> list[WorstCaseExample]:
    examples: list[WorstCaseExample] = []
    with target_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            examples.append(
                WorstCaseExample(
                    prompt_index=int(row["prompt_index"]),
                    example_key=row["example_key"],
                    prompt_text=row["prompt_text"],
                    prompt_tokens=int(row["prompt_tokens"]),
                    expected_worst_case_length=float(row["expected_worst_case_length"]),
                )
            )
    if not examples:
        raise ValueError(f"No examples found in {target_csv}")
    return examples


def assign_top_bottom_quartile_labels(
    examples: list[WorstCaseExample],
    low_quantile: float,
    high_quantile: float,
) -> dict[str, float]:
    values = np.array([example.expected_worst_case_length for example in examples], dtype=np.float64)
    low_threshold = float(np.quantile(values, low_quantile))
    high_threshold = float(np.quantile(values, high_quantile))
    for example in examples:
        if example.expected_worst_case_length <= low_threshold:
            example.label = 0
        elif example.expected_worst_case_length >= high_threshold:
            example.label = 1
        else:
            example.label = -1
    return {
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }


def build_splits(
    examples: list[WorstCaseExample],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[WorstCaseExample]]:
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
        "train": [example for example in examples if example.prompt_index in train_prompts and example.label >= 0],
        "val": [example for example in examples if example.prompt_index in val_prompts and example.label >= 0],
        "test": [example for example in examples if example.prompt_index in test_prompts and example.label >= 0],
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


def make_collate_fn(tokenizer, max_prompt_tokens: int):
    def collate(examples: list[WorstCaseExample]) -> dict[str, Any]:
        batch = tokenizer(
            [example.prompt_text for example in examples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        batch["labels"] = torch.tensor([example.label for example in examples], dtype=torch.float32)
        batch["prompt_tokens"] = torch.tensor([example.prompt_tokens for example in examples], dtype=torch.float32)
        batch["prompt_index"] = torch.tensor([example.prompt_index for example in examples], dtype=torch.long)
        batch["targets"] = torch.tensor(
            [example.expected_worst_case_length for example in examples],
            dtype=torch.float32,
        )
        batch["example_key"] = [example.example_key for example in examples]
        return batch

    return collate


def accuracy_from_probs(labels: np.ndarray, probs: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    preds = (probs >= 0.5).astype(np.int64)
    return float(np.mean(preds == labels))


def binary_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    pos = int(labels.sum())
    neg = int((1 - labels).sum())
    if pos == 0 or neg == 0:
        return 0.0

    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1, dtype=np.float64)

    sorted_probs = probs[order]
    start = 0
    while start < len(sorted_probs):
        end = start + 1
        while end < len(sorted_probs) and sorted_probs[end] == sorted_probs[start]:
            end += 1
        if end - start > 1:
            avg_rank = float(np.mean(ranks[order[start:end]]))
            ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = float(ranks[labels == 1].sum())
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels_list: list[int] = []
    probs_list: list[float] = []
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_examples += labels.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy().astype(np.int64)
            labels_list.extend(int(value) for value in labels_np)
            probs_list.extend(float(value) for value in probs)
            for idx in range(labels.size(0)):
                predictions.append(
                    {
                        "model_name": "lora_quartile_classifier",
                        "example_key": batch["example_key"][idx],
                        "prompt_index": int(batch["prompt_index"][idx]),
                        "prompt_tokens": float(batch["prompt_tokens"][idx]),
                        "expected_worst_case_length": float(batch["targets"][idx]),
                        "true_label": int(labels_np[idx]),
                        "predicted_probability": float(probs[idx]),
                        "predicted_label": int(probs[idx] >= 0.5),
                    }
                )

    labels_array = np.array(labels_list, dtype=np.int64)
    probs_array = np.array(probs_list, dtype=np.float64)
    metrics = {
        "count": total_examples,
        "loss": total_loss / total_examples if total_examples else 0.0,
        "accuracy": accuracy_from_probs(labels_array, probs_array),
        "auroc": binary_auroc(labels_array, probs_array),
        "positive_rate": float(labels_array.mean()) if total_examples else 0.0,
    }
    return metrics, predictions


def fit_prompt_tokens_baseline(
    train_examples: list[WorstCaseExample],
    eval_examples: list[WorstCaseExample],
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not train_examples or not eval_examples:
        return {"count": 0, "loss": 0.0, "accuracy": 0.0, "auroc": 0.0, "positive_rate": 0.0}, []

    train_x = np.array([[example.prompt_tokens] for example in train_examples], dtype=np.float32)
    train_y = np.array([example.label for example in train_examples], dtype=np.float32)
    eval_x = np.array([[example.prompt_tokens] for example in eval_examples], dtype=np.float32)
    eval_y = np.array([example.label for example in eval_examples], dtype=np.int64)

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = (train_x - mean) / std
    eval_x = (eval_x - mean) / std

    torch.manual_seed(seed)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    train_x_t = torch.from_numpy(train_x)
    train_y_t = torch.from_numpy(train_y).unsqueeze(1)

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    patience = 50
    stale = 0
    for _ in range(400):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x_t)
        loss = loss_fn(logits, train_y_t)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if loss_value + 1e-9 < best_loss:
            best_loss = loss_value
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        eval_probs = torch.sigmoid(model(torch.from_numpy(eval_x))).squeeze(1).numpy()

    predictions = []
    for example, prob in zip(eval_examples, eval_probs, strict=True):
        predictions.append(
            {
                "model_name": "prompt_tokens_baseline",
                "example_key": example.example_key,
                "prompt_index": example.prompt_index,
                "prompt_tokens": float(example.prompt_tokens),
                "expected_worst_case_length": float(example.expected_worst_case_length),
                "true_label": int(example.label),
                "predicted_probability": float(prob),
                "predicted_label": int(prob >= 0.5),
            }
        )

    metrics = {
        "count": len(eval_examples),
        "loss": 0.0,
        "accuracy": accuracy_from_probs(eval_y, eval_probs),
        "auroc": binary_auroc(eval_y, eval_probs),
        "positive_rate": float(eval_y.mean()) if len(eval_y) else 0.0,
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


def ensure_nonempty_split(splits: dict[str, list[WorstCaseExample]]) -> None:
    for name, subset in splits.items():
        labels = {example.label for example in subset}
        if not subset:
            raise ValueError(f"Split {name} has no labeled quartile examples.")
        if labels != {0, 1}:
            raise ValueError(f"Split {name} does not contain both quartile classes: labels={sorted(labels)}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    examples = load_examples(args.target_csv)
    thresholds = assign_top_bottom_quartile_labels(examples, args.low_quantile, args.high_quantile)
    splits = build_splits(examples, args.train_prompt_fraction, args.val_prompt_fraction, args.seed)
    ensure_nonempty_split(splits)

    tokenizer = make_tokenizer(args.base_model)
    device = torch.device(args.device)
    backbone = build_backbone(args.base_model, device)
    checkpoint_path = args.checkpoint_dir / f"global_step_{args.init_checkpoint_step}" / "actor" / "model_world_size_1_rank_0.pt"
    load_checkpoint_into_backbone(backbone, checkpoint_path)
    model = WorstCaseClassifier(backbone).to(device)

    collate_fn = make_collate_fn(tokenizer, args.max_prompt_tokens)
    train_loader = DataLoader(WorstCaseDataset(splits["train"]), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(WorstCaseDataset(splits["val"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(WorstCaseDataset(splits["test"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)

        val_metrics, _ = evaluate(model, val_loader, device)
        training_rows.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / seen if seen else 0.0,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_auroc": val_metrics["auroc"],
            }
        )
        print(
            f"[epoch {epoch}] train_loss={training_rows[-1]['train_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_auroc={val_metrics['auroc']:.3f}",
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
    baseline_test_metrics, baseline_test_predictions = fit_prompt_tokens_baseline(
        splits["train"],
        splits["test"],
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "adapter"
    model.backbone.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "dropout_p": model.dropout.p,
            "init_checkpoint_step": args.init_checkpoint_step,
            "thresholds": thresholds,
        },
        args.output_dir / "classifier_head.pt",
    )

    write_csv(training_rows, args.output_dir / "training_log.csv")
    write_csv(train_predictions, args.output_dir / "train_predictions.csv")
    write_csv(val_predictions, args.output_dir / "val_predictions.csv")
    write_csv(test_predictions, args.output_dir / "test_predictions.csv")
    write_csv(baseline_test_predictions, args.output_dir / "baseline_test_predictions.csv")

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "prompt_tokens_baseline_test": baseline_test_metrics,
        "thresholds": thresholds,
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
