#!/usr/bin/env python3
"""Collect per-checkpoint rollout snapshots across policy-grid runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--eval-files", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-checkpoints-per-policy", type=int, default=0, help="0 means all available checkpoints.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_examples(paths: list[Path], limit: int, seed: int) -> list[Example]:
    items: list[Example] = []
    for path in paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
            extra_info = row.get("extra_info") or {}
            data_source = row["data_source"]
            index = extra_info.get("index", -1)
            key = f"{data_source}::{index}"
            items.append(
                Example(
                    split="eval",
                    data_source=data_source,
                    prompt_messages=row["prompt"],
                    ground_truth=row["reward_model"]["ground_truth"],
                    extra_info=extra_info,
                    key=key,
                )
            )
    rng = random.Random(seed)
    rng.shuffle(items)
    if limit > 0:
        return items[:limit]
    return items


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


def load_checkpoint(model, step_dir: Path) -> None:
    ckpt = step_dir / "actor" / "model_world_size_1_rank_0.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def eval_examples(
    *,
    model,
    tokenizer,
    examples: list[Example],
    max_prompt_tokens: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eos_token_id = tokenizer.eos_token_id
    for ex in examples:
        prompt = render_prompt(tokenizer, ex.prompt_messages)
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )["input_ids"].to(device)
        attention_mask = torch.ones_like(ids)
        out = model.generate(
            input_ids=ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_k=0,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        response_ids = out[0, ids.shape[1] :].detach().cpu().tolist()
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
                "example_key": ex.key,
                "data_source": ex.data_source,
                "index": ex.extra_info.get("index"),
                "score": score,
                "solved": int(score >= 0.999999),
                "response_length": len(response_ids),
                "response_text": response_text,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_checkpoints(job: dict[str, Any], max_per_policy: int) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    job_type = job.get("job_type", "static_selector")
    if job_type == "periodic_gradient":
        periodic_root_raw = job.get("periodic_run_root")
        if periodic_root_raw:
            manifest = Path(periodic_root_raw) / "periodic_gradient_selector_manifest.json"
            if manifest.exists():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                for window in payload.get("windows", []):
                    ckpt = Path(window.get("checkpoint_path", ""))
                    step = int(window.get("end_step", -1))
                    if ckpt.exists() and step >= 0:
                        out.append((step, ckpt))
    else:
        run_dir = Path(job.get("run_dir", ""))
        if run_dir.exists():
            for ckpt in run_dir.glob("global_step_*"):
                if not ckpt.is_dir():
                    continue
                name = ckpt.name
                try:
                    step = int(name.split("global_step_", 1)[1])
                except Exception:
                    continue
                out.append((step, ckpt))
    out = sorted(out, key=lambda x: x[0])
    if max_per_policy > 0 and len(out) > max_per_policy:
        # keep latest checkpoints for temporal analysis
        out = out[-max_per_policy:]
    return out


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or (args.run_root / "policy_grid_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    examples = read_examples(args.eval_files, args.eval_limit, args.seed)
    if not examples:
        raise ValueError("No evaluation examples loaded.")

    output_dir = args.output_dir or (args.run_root / "analysis" / "rollout_snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = make_tokenizer(args.base_model)
    model = build_model(args.base_model, device)

    summary_rows: list[dict[str, Any]] = []
    for job in manifest.get("jobs", []):
        policy = job.get("policy", {})
        policy_name = policy.get("name")
        if not isinstance(policy_name, str):
            continue
        checkpoints = collect_checkpoints(job, args.max_checkpoints_per_policy)
        if not checkpoints:
            continue
        for step, ckpt_dir in checkpoints:
            load_checkpoint(model, ckpt_dir)
            rows = eval_examples(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            for row in rows:
                row["policy"] = policy_name
                row["global_step"] = step
                row["checkpoint_dir"] = str(ckpt_dir)
            out_csv = output_dir / policy_name / f"rollout_step_{step}.csv"
            write_csv(out_csv, rows)
            acc = sum(r["solved"] for r in rows) / len(rows)
            summary_rows.append(
                {
                    "policy": policy_name,
                    "global_step": step,
                    "checkpoint_dir": str(ckpt_dir),
                    "num_examples": len(rows),
                    "accuracy": acc,
                    "mean_response_length": sum(r["response_length"] for r in rows) / len(rows),
                    "output_csv": str(out_csv),
                }
            )
            print(f"wrote {out_csv} acc={acc:.4f}")

    write_csv(output_dir / "rollout_snapshot_index.csv", summary_rows)
    print(f"Wrote index: {output_dir / 'rollout_snapshot_index.csv'}")


if __name__ == "__main__":
    main()
