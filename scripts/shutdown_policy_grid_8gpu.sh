#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-runs/policy_grid_8gpu}"
MANIFEST_PATH="${2:-${RUN_ROOT}/policy_grid_manifest.json}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

echo "Run root: ${RUN_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"

mapfile -t PID_LINES < <(
  python3 - "${MANIFEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for job in manifest.get("jobs", []):
    pid = int(job.get("pid", -1))
    name = job.get("policy", {}).get("name", "unknown")
    if pid > 0:
        print(f"{pid}\t{name}")
PY
)

if [[ "${#PID_LINES[@]}" -eq 0 ]]; then
  echo "No valid PIDs in manifest."
else
  echo "Stopping policy jobs..."
  for line in "${PID_LINES[@]}"; do
    pid="${line%%$'\t'*}"
    policy="${line#*$'\t'}"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" || true
      echo "  SIGTERM pid=${pid} policy=${policy}"
    else
      echo "  already stopped pid=${pid} policy=${policy}"
    fi
  done
fi

# Best-effort stop of leftover launcher/trainer processes for this run root.
pkill -f "run_policy_grid_8gpu.py --run-root ${RUN_ROOT}" 2>/dev/null || true
pkill -f "${RUN_ROOT}/policies/" 2>/dev/null || true

echo "Done. Remaining matching processes:"
ps -eo pid,etime,cmd | rg -i "${RUN_ROOT}|run_policy_grid_8gpu|run_grpo_gradient_selector_continuation|verl.trainer.main_ppo" | rg -v rg || true
