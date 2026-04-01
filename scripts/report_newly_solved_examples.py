#!/usr/bin/env python3
"""Compare two checkpoints and report newly solved / forgotten examples."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score import default_compute_score


@dataclass
class Example:
    split: str
    data_source: str
    prompt_messages: list[dict[str, str]]
    ground_truth: str
    extra_info: dict[str, Any]
    key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-step-dir", type=Path, required=True, help="Path to global_step_* directory (baseline).")
    parser.add_argument("--end-step-dir", type=Path, required=True, help="Path to global_step_* directory (target).")
    parser.add_argument(
        "--train-files",
        type=Path,
        nargs="+",
        required=True,
        help="Train parquet files to evaluate.",
    )
    parser.add_argument(
        "--val-files",
        type=Path,
        nargs="+",
        required=True,
        help="Validation parquet files to evaluate.",
    )
    parser.add_argument("--train-limit", type=int, default=0, help="0 means all.")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means all.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_examples(paths: list[Path], *, split: str, limit: int) -> list[Example]:
    examples: list[Example] = []
    for path in paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
            extra_info = row.get("extra_info") or {}
            data_source = row["data_source"]
            index = extra_info.get("index", -1)
            key = f"{split}::{data_source}::{index}"
            examples.append(
                Example(
                    split=split,
                    data_source=data_source,
                    prompt_messages=row["prompt"],
                    ground_truth=row["reward_model"]["ground_truth"],
                    extra_info=extra_info,
                    key=key,
                )
            )
            if limit > 0 and len(examples) >= limit:
                return examples
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


def checkpoint_model_path(step_dir: Path) -> Path:
    return step_dir / "actor" / "model_world_size_1_rank_0.pt"


def load_checkpoint(model, step_dir: Path) -> None:
    ckpt = checkpoint_model_path(step_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch for {ckpt}: missing={missing[:5]} unexpected={unexpected[:5]}"
        )


def render_prompt(tokenizer, prompt_messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def evaluate_examples(
    *,
    model,
    tokenizer,
    examples: list[Example],
    max_prompt_tokens: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    eos_token_id = tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []
    for ex in examples:
        prompt_text = render_prompt(tokenizer, ex.prompt_messages)
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )["input_ids"].to(device)
        attention_mask = torch.ones_like(prompt_ids)
        output = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_k=0,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        response_ids = output[0, prompt_ids.shape[1] :].detach().cpu().tolist()
        if eos_token_id is not None and eos_token_id in response_ids:
            response_ids = response_ids[: response_ids.index(eos_token_id)]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        score = float(
            default_compute_score(
                ex.data_source,
                response_text,
                ex.ground_truth,
                extra_info=ex.extra_info,
            )
        )
        rows.append(
            {
                "key": ex.key,
                "split": ex.split,
                "data_source": ex.data_source,
                "index": ex.extra_info.get("index"),
                "score": score,
                "solved": int(score >= 0.999999),
                "response_text": response_text,
            }
        )
    return rows


def summarize_delta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["data_source"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (split, source), bucket in sorted(grouped.items()):
        acc_start = sum(r["start_solved"] for r in bucket) / len(bucket)
        acc_end = sum(r["end_solved"] for r in bucket) / len(bucket)
        summaries.append(
            {
                "split": split,
                "data_source": source,
                "count": len(bucket),
                "acc_start": acc_start,
                "acc_end": acc_end,
                "acc_delta": acc_end - acc_start,
                "newly_solved": sum(1 for r in bucket if r["newly_solved"] == 1),
                "forgotten": sum(1 for r in bucket if r["forgotten"] == 1),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = read_examples(args.train_files, split="train", limit=args.train_limit)
    val_examples = read_examples(args.val_files, split="val", limit=args.val_limit)
    examples = train_examples + val_examples
    if not examples:
        raise ValueError("No examples loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = make_tokenizer(args.base_model)
    model = build_model(args.base_model, device)

    load_checkpoint(model, args.start_step_dir)
    start_rows = evaluate_examples(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    start_map = {row["key"]: row for row in start_rows}
    write_csv(args.output_dir / "start_predictions.csv", start_rows)

    load_checkpoint(model, args.end_step_dir)
    end_rows = evaluate_examples(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    end_map = {row["key"]: row for row in end_rows}
    write_csv(args.output_dir / "end_predictions.csv", end_rows)

    delta_rows: list[dict[str, Any]] = []
    newly_solved_rows: list[dict[str, Any]] = []
    for ex in examples:
        s = start_map[ex.key]
        e = end_map[ex.key]
        row = {
            "key": ex.key,
            "split": ex.split,
            "data_source": ex.data_source,
            "index": ex.extra_info.get("index"),
            "start_score": s["score"],
            "end_score": e["score"],
            "start_solved": s["solved"],
            "end_solved": e["solved"],
            "score_delta": e["score"] - s["score"],
            "newly_solved": int(s["solved"] == 0 and e["solved"] == 1),
            "forgotten": int(s["solved"] == 1 and e["solved"] == 0),
            "start_response_text": s["response_text"],
            "end_response_text": e["response_text"],
        }
        delta_rows.append(row)
        if row["newly_solved"] == 1:
            newly_solved_rows.append(row)

    write_csv(args.output_dir / "delta_per_example.csv", delta_rows)
    write_csv(args.output_dir / "newly_solved_examples.csv", newly_solved_rows)

    summary_rows = summarize_delta(delta_rows)
    write_csv(args.output_dir / "summary_by_split_source.csv", summary_rows)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "start_step_dir": str(args.start_step_dir),
                "end_step_dir": str(args.end_step_dir),
                "train_files": [str(p) for p in args.train_files],
                "val_files": [str(p) for p in args.val_files],
                "train_count": len(train_examples),
                "val_count": len(val_examples),
                "max_prompt_tokens": args.max_prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {args.output_dir / 'delta_per_example.csv'}")
    print(f"Wrote: {args.output_dir / 'newly_solved_examples.csv'}")
    print(f"Wrote: {args.output_dir / 'summary_by_split_source.csv'}")


if __name__ == "__main__":
    main()
