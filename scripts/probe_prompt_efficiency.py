#!/usr/bin/env python3
"""Probe whether prompt-only checkpoint representations predict efficiency targets.

This script freezes each profiled checkpoint, encodes the prompt with the
matching checkpoint, and trains a small regression head on top of the prompt
representation. Evaluation uses prompt-held-out cross-validation so the probe
cannot memorize the same prompt across checkpoints.
"""

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
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_COLUMNS = (
    "gradient_statistical_efficiency",
    "computational_efficiency",
)


@dataclass
class ProbeExample:
    checkpoint_step: int
    prompt_index: int
    example_key: str
    prompt_text: str
    prompt_tokens: int
    targets: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=Path("outputs/profiling_gsm8k_efficiency_2026-03-27_v32_len1024/ground_truth_profile.csv"),
    )
    parser.add_argument(
        "--rollout-inspection-csv",
        type=Path,
        default=Path("outputs/profiling_gsm8k_efficiency_2026-03-27_v32_len1024/rollout_inspection.csv"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/EffiRL/qwen2.5-1.5b-grpo-math-single-gpu-vllm"),
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200],
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=0, help="0 means linear head.")
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--classification-low-quantile", type=float, default=0.25)
    parser.add_argument("--classification-high-quantile", type=float, default=0.75)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/profiling_gsm8k_efficiency_2026-03-27_v32_len1024/analysis/prompt_probe"),
    )
    return parser.parse_args()


def read_prompt_texts(path: Path) -> dict[str, str]:
    prompt_text_by_key: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row["example_key"]
            prompt_text = row["prompt_text"]
            existing = prompt_text_by_key.get(key)
            if existing is None:
                prompt_text_by_key[key] = prompt_text
            elif existing != prompt_text:
                raise ValueError(f"Inconsistent prompt text for example_key={key}")
    return prompt_text_by_key


def read_examples(profile_csv: Path, prompt_text_by_key: dict[str, str], checkpoint_steps: set[int]) -> list[ProbeExample]:
    examples: list[ProbeExample] = []
    with profile_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            checkpoint_step = int(row["checkpoint_step"])
            if checkpoint_step not in checkpoint_steps:
                continue
            key = row["example_key"]
            prompt_text = prompt_text_by_key.get(key)
            if prompt_text is None:
                raise KeyError(f"Missing prompt_text for example_key={key}")
            targets = {column: float(row[column]) for column in TARGET_COLUMNS}
            examples.append(
                ProbeExample(
                    checkpoint_step=checkpoint_step,
                    prompt_index=int(row["prompt_index"]),
                    example_key=key,
                    prompt_text=prompt_text,
                    prompt_tokens=int(row["prompt_tokens"]),
                    targets=targets,
                )
            )
    if not examples:
        raise ValueError("No examples loaded for the requested checkpoint steps.")
    return examples


def make_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_model(base_model: str, device: torch.device):
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
    model.eval()
    return model


def load_checkpoint_into_model(model, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load checkpoint cleanly: {checkpoint_path}\nmissing={missing[:10]}\nunexpected={unexpected[:10]}"
        )


def encode_prompts(
    model,
    tokenizer,
    prompts: list[str],
    max_prompt_tokens: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            last_hidden = outputs.hidden_states[-1]
            last_token_index = encoded["attention_mask"].sum(dim=1) - 1
            pooled = last_hidden[torch.arange(last_hidden.size(0), device=device), last_token_index]
            vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def build_group_folds(prompt_indices: list[int], folds: int, seed: int) -> list[set[int]]:
    unique_prompts = sorted(set(prompt_indices))
    if folds <= 1 or folds > len(unique_prompts):
        raise ValueError(f"folds must be in [2, {len(unique_prompts)}], got {folds}.")
    rng = random.Random(seed)
    rng.shuffle(unique_prompts)
    buckets = [set() for _ in range(folds)]
    for idx, prompt_index in enumerate(unique_prompts):
        buckets[idx % folds].add(prompt_index)
    return buckets


def choose_validation_prompts(train_prompts: list[int], seed: int) -> set[int]:
    unique_prompts = sorted(set(train_prompts))
    if len(unique_prompts) <= 1:
        return set(unique_prompts)
    rng = random.Random(seed)
    rng.shuffle(unique_prompts)
    count = max(1, round(0.2 * len(unique_prompts)))
    count = min(count, len(unique_prompts) - 1)
    return set(unique_prompts[:count])


def make_probe(input_dim: int, hidden_dim: int) -> nn.Module:
    if hidden_dim > 0:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    return nn.Linear(input_dim, 1)


def standardize_target(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean())
    std = float(train_y.std())
    if std < 1e-6:
        std = 1.0
    return (train_y - mean) / std, (test_y - mean) / std, mean, std


def train_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> nn.Module:
    device = torch.device("cpu")
    model = make_probe(train_x.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_x_t = torch.from_numpy(train_x).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().unsqueeze(1).to(device)
    val_x_t = torch.from_numpy(val_x).float().to(device)
    val_y_t = torch.from_numpy(val_y).float().unsqueeze(1).to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(train_x_t)
        loss = loss_fn(pred, train_y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x_t), val_y_t).item())
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


def train_binary_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> nn.Module:
    device = torch.device("cpu")
    model = make_probe(train_x.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_x_t = torch.from_numpy(train_x).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().unsqueeze(1).to(device)
    val_x_t = torch.from_numpy(val_x).float().to(device)
    val_y_t = torch.from_numpy(val_y).float().unsqueeze(1).to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x_t)
        loss = loss_fn(logits, train_y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x_t), val_y_t).item())
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_probe(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        tensor = torch.from_numpy(x).float()
        pred = model(tensor).squeeze(1).cpu().numpy()
    return pred


def predict_binary_probe(model: nn.Module, x: np.ndarray) -> np.ndarray:
    logits = predict_probe(model, x)
    return 1.0 / (1.0 + np.exp(-logits))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_std = float(y_true.std())
    pred_std = float(y_pred.std())
    if true_std < 1e-12 or pred_std < 1e-12:
        return 0.0
    centered_true = y_true - y_true.mean()
    centered_pred = y_pred - y_pred.mean()
    return float(np.mean(centered_true * centered_pred) / (true_std * pred_std))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    tpr = float(np.mean(y_pred[positives] == 1)) if positives.any() else 0.0
    tnr = float(np.mean(y_pred[negatives] == 0)) if negatives.any() else 0.0
    return 0.5 * (tpr + tnr)


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_score), dtype=np.float64) + 1.0
    pos_ranks = ranks[positives]
    u_stat = float(pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0)
    return u_stat / (n_pos * n_neg)


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "count": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "pearson_r": 0.0,
        }
    y_true = np.array([float(row["true_value"]) for row in rows], dtype=np.float64)
    y_pred = np.array([float(row["predicted_value"]) for row in rows], dtype=np.float64)
    return {
        "count": float(len(rows)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson_r": pearsonr(y_true, y_pred),
    }


def evaluate_classification_predictions(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "count": 0.0,
            "positive_rate": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "roc_auc": 0.5,
        }
    y_true = np.array([int(row["true_label"]) for row in rows], dtype=np.int64)
    y_prob = np.array([float(row["predicted_probability"]) for row in rows], dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return {
        "count": float(len(rows)),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def run_grouped_probe(
    examples: list[ProbeExample],
    features: np.ndarray,
    checkpoint_steps: list[int],
    target_column: str,
    model_name: str,
    folds: int,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> list[dict[str, Any]]:
    checkpoint_index = {step: idx for idx, step in enumerate(checkpoint_steps)}
    step_onehot = np.zeros((len(examples), len(checkpoint_steps)), dtype=np.float32)
    for idx, example in enumerate(examples):
        step_onehot[idx, checkpoint_index[example.checkpoint_step]] = 1.0

    prompt_indices = [example.prompt_index for example in examples]
    target_values = np.array([example.targets[target_column] for example in examples], dtype=np.float32)
    fold_prompt_sets = build_group_folds(prompt_indices, folds, seed)
    predictions: list[dict[str, Any]] = []

    for fold_index, test_prompts in enumerate(fold_prompt_sets):
        train_mask = np.array([prompt not in test_prompts for prompt in prompt_indices], dtype=bool)
        test_mask = ~train_mask
        train_prompts = [prompt_indices[idx] for idx in range(len(prompt_indices)) if train_mask[idx]]
        val_prompts = choose_validation_prompts(train_prompts, seed + 1000 + fold_index)
        inner_train_mask = np.array([train_mask[idx] and prompt_indices[idx] not in val_prompts for idx in range(len(prompt_indices))], dtype=bool)
        val_mask = np.array([train_mask[idx] and prompt_indices[idx] in val_prompts for idx in range(len(prompt_indices))], dtype=bool)

        full_features = np.concatenate([features, step_onehot], axis=1)
        train_x_raw = full_features[inner_train_mask]
        val_x_raw = full_features[val_mask]
        test_x_raw = full_features[test_mask]

        train_mean = train_x_raw.mean(axis=0, keepdims=True)
        train_std = train_x_raw.std(axis=0, keepdims=True)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        train_x = (train_x_raw - train_mean) / train_std
        val_x = (val_x_raw - train_mean) / train_std
        test_x = (test_x_raw - train_mean) / train_std

        train_y_raw = target_values[inner_train_mask]
        val_y_raw = target_values[val_mask]
        test_y_raw = target_values[test_mask]
        train_y, val_y, train_y_mean, train_y_std = standardize_target(train_y_raw, val_y_raw)

        model = train_probe(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        pred_y = predict_probe(model, test_x) * train_y_std + train_y_mean

        test_indices = np.where(test_mask)[0]
        for local_idx, example_idx in enumerate(test_indices):
            example = examples[example_idx]
            predictions.append(
                {
                    "model_name": model_name,
                    "target_column": target_column,
                    "fold": fold_index,
                    "checkpoint_step": example.checkpoint_step,
                    "prompt_index": example.prompt_index,
                    "example_key": example.example_key,
                    "true_value": float(test_y_raw[local_idx]),
                    "predicted_value": float(pred_y[local_idx]),
                    "abs_error": float(abs(pred_y[local_idx] - test_y_raw[local_idx])),
                }
            )
    return predictions


def run_grouped_quartile_classification(
    examples: list[ProbeExample],
    features: np.ndarray,
    checkpoint_steps: list[int],
    target_column: str,
    model_name: str,
    folds: int,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    low_quantile: float,
    high_quantile: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    checkpoint_index = {step: idx for idx, step in enumerate(checkpoint_steps)}
    step_onehot = np.zeros((len(examples), len(checkpoint_steps)), dtype=np.float32)
    for idx, example in enumerate(examples):
        step_onehot[idx, checkpoint_index[example.checkpoint_step]] = 1.0

    target_values = np.array([example.targets[target_column] for example in examples], dtype=np.float32)
    low_threshold = float(np.quantile(target_values, low_quantile))
    high_threshold = float(np.quantile(target_values, high_quantile))
    labels = np.full(len(examples), -1, dtype=np.int64)
    labels[target_values <= low_threshold] = 0
    labels[target_values >= high_threshold] = 1

    labeled_indices = np.where(labels >= 0)[0]
    labeled_prompt_indices = [examples[idx].prompt_index for idx in labeled_indices]
    fold_prompt_sets = build_group_folds(labeled_prompt_indices, folds, seed)

    predictions: list[dict[str, Any]] = []
    full_features = np.concatenate([features, step_onehot], axis=1)
    for fold_index, test_prompts in enumerate(fold_prompt_sets):
        labeled_examples = [idx for idx in labeled_indices if examples[idx].prompt_index not in test_prompts]
        test_examples = [idx for idx in labeled_indices if examples[idx].prompt_index in test_prompts]
        train_prompts = [examples[idx].prompt_index for idx in labeled_examples]
        val_prompts = choose_validation_prompts(train_prompts, seed + 2000 + fold_index)

        inner_train_indices = [idx for idx in labeled_examples if examples[idx].prompt_index not in val_prompts]
        val_indices = [idx for idx in labeled_examples if examples[idx].prompt_index in val_prompts]

        train_x_raw = full_features[inner_train_indices]
        val_x_raw = full_features[val_indices]
        test_x_raw = full_features[test_examples]

        train_mean = train_x_raw.mean(axis=0, keepdims=True)
        train_std = train_x_raw.std(axis=0, keepdims=True)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        train_x = (train_x_raw - train_mean) / train_std
        val_x = (val_x_raw - train_mean) / train_std
        test_x = (test_x_raw - train_mean) / train_std

        train_y = labels[inner_train_indices].astype(np.float32)
        val_y = labels[val_indices].astype(np.float32)
        test_y = labels[test_examples].astype(np.int64)

        model = train_binary_probe(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        pred_prob = predict_binary_probe(model, test_x)

        for local_idx, example_idx in enumerate(test_examples):
            example = examples[example_idx]
            predictions.append(
                {
                    "model_name": model_name,
                    "target_column": target_column,
                    "fold": fold_index,
                    "checkpoint_step": example.checkpoint_step,
                    "prompt_index": example.prompt_index,
                    "example_key": example.example_key,
                    "true_label": int(test_y[local_idx]),
                    "predicted_probability": float(pred_prob[local_idx]),
                    "predicted_label": int(pred_prob[local_idx] >= 0.5),
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                }
            )

    thresholds = {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
    }
    return predictions, thresholds


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_predictions(predictions: list[dict[str, Any]], checkpoint_steps: list[int]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    model_names = sorted({row["model_name"] for row in predictions})
    target_columns = sorted({row["target_column"] for row in predictions})

    for model_name in model_names:
        for target_column in target_columns:
            subset = [row for row in predictions if row["model_name"] == model_name and row["target_column"] == target_column]
            overall = evaluate_predictions(subset)
            summary_rows.append(
                {
                    "scope": "overall",
                    "model_name": model_name,
                    "target_column": target_column,
                    "checkpoint_step": "all",
                    **overall,
                }
            )
            for checkpoint_step in checkpoint_steps:
                per_step = [row for row in subset if int(row["checkpoint_step"]) == checkpoint_step]
                metrics = evaluate_predictions(per_step)
                summary_rows.append(
                    {
                        "scope": "per_checkpoint",
                        "model_name": model_name,
                        "target_column": target_column,
                        "checkpoint_step": checkpoint_step,
                        **metrics,
                    }
                )
    return summary_rows


def summarize_classification_predictions(predictions: list[dict[str, Any]], checkpoint_steps: list[int]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    model_names = sorted({row["model_name"] for row in predictions})
    target_columns = sorted({row["target_column"] for row in predictions})

    for model_name in model_names:
        for target_column in target_columns:
            subset = [row for row in predictions if row["model_name"] == model_name and row["target_column"] == target_column]
            overall = evaluate_classification_predictions(subset)
            threshold_row = subset[0] if subset else {}
            summary_rows.append(
                {
                    "scope": "overall",
                    "model_name": model_name,
                    "target_column": target_column,
                    "checkpoint_step": "all",
                    "low_threshold": threshold_row.get("low_threshold", ""),
                    "high_threshold": threshold_row.get("high_threshold", ""),
                    **overall,
                }
            )
            for checkpoint_step in checkpoint_steps:
                per_step = [row for row in subset if int(row["checkpoint_step"]) == checkpoint_step]
                metrics = evaluate_classification_predictions(per_step)
                summary_rows.append(
                    {
                        "scope": "per_checkpoint",
                        "model_name": model_name,
                        "target_column": target_column,
                        "checkpoint_step": checkpoint_step,
                        "low_threshold": threshold_row.get("low_threshold", ""),
                        "high_threshold": threshold_row.get("high_threshold", ""),
                        **metrics,
                    }
                )
    return summary_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_steps = list(args.checkpoint_steps)
    prompt_text_by_key = read_prompt_texts(args.rollout_inspection_csv)
    examples = read_examples(args.profile_csv, prompt_text_by_key, set(checkpoint_steps))
    examples.sort(key=lambda ex: (ex.checkpoint_step, ex.prompt_index))

    tokenizer = make_tokenizer(args.base_model)
    device = torch.device(args.device)
    model = build_model(args.base_model, device)

    feature_blocks: list[np.ndarray] = []
    for checkpoint_step in checkpoint_steps:
        checkpoint_path = args.checkpoint_dir / f"global_step_{checkpoint_step}" / "actor" / "model_world_size_1_rank_0.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        load_checkpoint_into_model(model, checkpoint_path)
        step_examples = [example for example in examples if example.checkpoint_step == checkpoint_step]
        prompts = [example.prompt_text for example in step_examples]
        step_features = encode_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_tokens=args.max_prompt_tokens,
            batch_size=args.batch_size,
            device=device,
        )
        feature_blocks.append(step_features)
    checkpoint_probe_features = np.concatenate(feature_blocks, axis=0).astype(np.float32)

    length_features = np.array([[float(example.prompt_tokens)] for example in examples], dtype=np.float32)

    regression_predictions: list[dict[str, Any]] = []
    classification_predictions: list[dict[str, Any]] = []
    classification_thresholds: dict[str, dict[str, float]] = {}
    for target_column in TARGET_COLUMNS:
        regression_predictions.extend(
            run_grouped_probe(
                examples=examples,
                features=length_features,
                checkpoint_steps=checkpoint_steps,
                target_column=target_column,
                model_name="prompt_tokens_baseline",
                folds=args.folds,
                seed=args.seed,
                hidden_dim=0,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
            )
        )
        regression_predictions.extend(
            run_grouped_probe(
                examples=examples,
                features=checkpoint_probe_features,
                checkpoint_steps=checkpoint_steps,
                target_column=target_column,
                model_name="checkpoint_prompt_probe",
                folds=args.folds,
                seed=args.seed,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
            )
        )
        baseline_classification_rows, thresholds = run_grouped_quartile_classification(
            examples=examples,
            features=length_features,
            checkpoint_steps=checkpoint_steps,
            target_column=target_column,
            model_name="prompt_tokens_baseline",
            folds=args.folds,
            seed=args.seed,
            hidden_dim=0,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            low_quantile=args.classification_low_quantile,
            high_quantile=args.classification_high_quantile,
        )
        classification_predictions.extend(baseline_classification_rows)
        probe_classification_rows, thresholds = run_grouped_quartile_classification(
            examples=examples,
            features=checkpoint_probe_features,
            checkpoint_steps=checkpoint_steps,
            target_column=target_column,
            model_name="checkpoint_prompt_probe",
            folds=args.folds,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            low_quantile=args.classification_low_quantile,
            high_quantile=args.classification_high_quantile,
        )
        classification_predictions.extend(probe_classification_rows)
        classification_thresholds[target_column] = thresholds

    summary_rows = summarize_predictions(regression_predictions, checkpoint_steps)
    classification_summary_rows = summarize_classification_predictions(classification_predictions, checkpoint_steps)
    write_csv(regression_predictions, args.output_dir / "prompt_probe_predictions.csv")
    write_csv(summary_rows, args.output_dir / "prompt_probe_summary.csv")
    write_csv(classification_predictions, args.output_dir / "prompt_probe_quartile_predictions.csv")
    write_csv(classification_summary_rows, args.output_dir / "prompt_probe_quartile_summary.csv")

    manifest = {
        "profile_csv": str(args.profile_csv),
        "rollout_inspection_csv": str(args.rollout_inspection_csv),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_steps": checkpoint_steps,
        "base_model": args.base_model,
        "example_count": len(examples),
        "unique_prompt_count": len({example.prompt_index for example in examples}),
        "folds": args.folds,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "patience": args.patience,
        "device": str(device),
        "feature_dim": int(checkpoint_probe_features.shape[1]),
        "classification_low_quantile": args.classification_low_quantile,
        "classification_high_quantile": args.classification_high_quantile,
        "classification_thresholds": classification_thresholds,
    }
    (args.output_dir / "prompt_probe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_dir / 'prompt_probe_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'prompt_probe_summary.csv'}")
    print(f"Wrote {args.output_dir / 'prompt_probe_quartile_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'prompt_probe_quartile_summary.csv'}")
    print(f"Wrote {args.output_dir / 'prompt_probe_manifest.json'}")


if __name__ == "__main__":
    main()
