#!/usr/bin/env python3
"""Train a LoRA regressor for expected worst-case generation length."""

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


class WorstCaseDataset(Dataset):
    def __init__(self, examples: list[WorstCaseExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> WorstCaseExample:
        return self.examples[index]


class WorstCaseRegressor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(hidden_size, 1)

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
        return self.regressor(pooled.float()).squeeze(-1)


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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/worst_case_length_pool_gsm8k_ckpt200/regressor"),
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


def build_splits(examples: list[WorstCaseExample], train_fraction: float, val_fraction: float, seed: int) -> dict[str, list[WorstCaseExample]]:
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


def make_collate_fn(tokenizer, max_prompt_tokens: int):
    def collate(examples: list[WorstCaseExample]) -> dict[str, Any]:
        batch = tokenizer(
            [example.prompt_text for example in examples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        batch["targets"] = torch.tensor([example.expected_worst_case_length for example in examples], dtype=torch.float32)
        batch["prompt_tokens"] = torch.tensor([example.prompt_tokens for example in examples], dtype=torch.float32)
        batch["prompt_index"] = torch.tensor([example.prompt_index for example in examples], dtype=torch.long)
        batch["example_key"] = [example.example_key for example in examples]
        return batch

    return collate


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    y_true_std = float(y_true.std())
    y_pred_std = float(y_pred.std())
    if y_true_std < 1e-12 or y_pred_std < 1e-12:
        return 0.0
    centered_true = y_true - y_true.mean()
    centered_pred = y_pred - y_pred.mean()
    return float(np.mean(centered_true * centered_pred) / (y_true_std * y_pred_std))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    losses: list[float] = []
    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)
            normalized_targets = (targets - target_mean) / target_std
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pred_norm = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.mse_loss(pred_norm, normalized_targets)
            losses.append(float(loss.item()) * targets.size(0))

            pred = pred_norm * target_std + target_mean
            y_true = targets.cpu().numpy()
            y_pred = pred.cpu().numpy()
            y_true_list.extend(float(value) for value in y_true)
            y_pred_list.extend(float(value) for value in y_pred)

            for idx in range(targets.size(0)):
                predictions.append(
                    {
                        "example_key": batch["example_key"][idx],
                        "prompt_index": int(batch["prompt_index"][idx]),
                        "prompt_tokens": float(batch["prompt_tokens"][idx]),
                        "true_value": float(y_true[idx]),
                        "predicted_value": float(y_pred[idx]),
                        "abs_error": float(abs(y_pred[idx] - y_true[idx])),
                    }
                )

    y_true_array = np.array(y_true_list, dtype=np.float64)
    y_pred_array = np.array(y_pred_list, dtype=np.float64)
    metrics = {
        "count": len(y_true_list),
        "loss": sum(losses) / len(y_true_list) if y_true_list else 0.0,
        "mae": mae(y_true_array, y_pred_array) if y_true_list else 0.0,
        "rmse": rmse(y_true_array, y_pred_array) if y_true_list else 0.0,
        "pearson_r": pearsonr(y_true_array, y_pred_array) if y_true_list else 0.0,
        "r2": r2_score(y_true_array, y_pred_array) if y_true_list else 0.0,
    }
    return metrics, predictions


def fit_prompt_tokens_baseline(
    train_examples: list[WorstCaseExample],
    eval_examples: list[WorstCaseExample],
) -> dict[str, float]:
    train_x = np.array([example.prompt_tokens for example in train_examples], dtype=np.float64)
    train_y = np.array([example.expected_worst_case_length for example in train_examples], dtype=np.float64)
    eval_x = np.array([example.prompt_tokens for example in eval_examples], dtype=np.float64)
    eval_y = np.array([example.expected_worst_case_length for example in eval_examples], dtype=np.float64)

    design = np.stack([np.ones_like(train_x), train_x], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    eval_design = np.stack([np.ones_like(eval_x), eval_x], axis=1)
    pred_y = eval_design @ coeffs
    return {
        "mae": mae(eval_y, pred_y),
        "rmse": rmse(eval_y, pred_y),
        "pearson_r": pearsonr(eval_y, pred_y),
        "r2": r2_score(eval_y, pred_y),
    }


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

    examples = load_examples(args.target_csv)
    splits = build_splits(examples, args.train_prompt_fraction, args.val_prompt_fraction, args.seed)

    target_mean = float(np.mean([example.expected_worst_case_length for example in splits["train"]]))
    target_std = float(np.std([example.expected_worst_case_length for example in splits["train"]]))
    if target_std < 1e-6:
        target_std = 1.0

    tokenizer = make_tokenizer(args.base_model)
    device = torch.device(args.device)
    backbone = build_backbone(args.base_model, device)
    checkpoint_path = args.checkpoint_dir / f"global_step_{args.init_checkpoint_step}" / "actor" / "model_world_size_1_rank_0.pt"
    load_checkpoint_into_backbone(backbone, checkpoint_path)
    model = WorstCaseRegressor(backbone).to(device)

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
            targets = batch["targets"].to(device)
            normalized_targets = (targets - target_mean) / target_std
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.mse_loss(pred_norm, normalized_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            running_loss += float(loss.item()) * targets.size(0)
            seen += targets.size(0)

        val_metrics, _ = evaluate(model, val_loader, device, target_mean, target_std)
        training_rows.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / seen if seen else 0.0,
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_pearson_r": val_metrics["pearson_r"],
                "val_r2": val_metrics["r2"],
            }
        )
        print(
            f"[epoch {epoch}] train_loss={training_rows[-1]['train_loss']:.4f} "
            f"val_mae={val_metrics['mae']:.2f} val_rmse={val_metrics['rmse']:.2f} "
            f"val_r={val_metrics['pearson_r']:.3f}",
            flush=True,
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)

    train_metrics, train_predictions = evaluate(model, train_loader, device, target_mean, target_std)
    val_metrics, val_predictions = evaluate(model, val_loader, device, target_mean, target_std)
    test_metrics, test_predictions = evaluate(model, test_loader, device, target_mean, target_std)
    baseline_test_metrics = fit_prompt_tokens_baseline(splits["train"], splits["test"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "adapter"
    model.backbone.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(
        {
            "regressor_state_dict": model.regressor.state_dict(),
            "dropout_p": model.dropout.p,
            "target_mean": target_mean,
            "target_std": target_std,
            "init_checkpoint_step": args.init_checkpoint_step,
        },
        args.output_dir / "regressor_head.pt",
    )

    write_csv(training_rows, args.output_dir / "training_log.csv")
    write_csv(train_predictions, args.output_dir / "train_predictions.csv")
    write_csv(val_predictions, args.output_dir / "val_predictions.csv")
    write_csv(test_predictions, args.output_dir / "test_predictions.csv")

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "prompt_tokens_baseline_test": baseline_test_metrics,
        "train_examples": len(splits["train"]),
        "val_examples": len(splits["val"]),
        "test_examples": len(splits["test"]),
        "train_prompt_count": len({example.prompt_index for example in splits["train"]}),
        "val_prompt_count": len({example.prompt_index for example in splits["val"]}),
        "test_prompt_count": len({example.prompt_index for example in splits["test"]}),
        "target_mean": target_mean,
        "target_std": target_std,
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
