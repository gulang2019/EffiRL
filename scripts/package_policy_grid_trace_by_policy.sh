#!/usr/bin/env bash
set -euo pipefail

# Create per-policy trace archives for easier download/chunking.
#
# Usage:
#   bash scripts/package_policy_grid_trace_by_policy.sh runs/policy_grid_8gpu
#   bash scripts/package_policy_grid_trace_by_policy.sh runs/policy_grid_8gpu /tmp/policy_zips

RUN_ROOT="${1:-runs/policy_grid_8gpu}"
OUT_DIR="${2:-${RUN_ROOT%/}_trace_by_policy}"
MANIFEST="${RUN_ROOT}/policy_grid_manifest.json"

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Run root not found: ${RUN_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

# 1) Shared lightweight archive
SHARED_STAGE="${TMP_DIR}/shared"
mkdir -p "${SHARED_STAGE}/${RUN_ROOT}"
for f in \
  "${RUN_ROOT}/policy_grid_manifest.json" \
  "${RUN_ROOT}/profile.log"; do
  if [[ -f "${f}" ]]; then
    mkdir -p "${SHARED_STAGE}/$(dirname "${f}")"
    cp "${f}" "${SHARED_STAGE}/${f}"
  fi
done
for d in \
  "${RUN_ROOT}/progress"; do
  if [[ -d "${d}" ]]; then
    mkdir -p "${SHARED_STAGE}/$(dirname "${d}")"
    cp -r "${d}" "${SHARED_STAGE}/${d}"
  fi
done
(
  cd "${SHARED_STAGE}"
  zip -r "${OLDPWD}/${OUT_DIR}/shared_trace.zip" . >/dev/null
)

# 2) Per-policy archives
POLICIES=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && POLICIES+=("${line}")
done < <(
  python3 - "${MANIFEST}" <<'PY'
import json,sys
manifest=json.load(open(sys.argv[1],encoding="utf-8"))
names=[]
for job in manifest.get("jobs",[]):
    p=job.get("policy",{}).get("name")
    if isinstance(p,str):
        names.append(p)
for p in sorted(set(names)):
    print(p)
PY
)

for policy in "${POLICIES[@]}"; do
  STAGE="${TMP_DIR}/policy_${policy}"
  mkdir -p "${STAGE}"

  # policy run logs/manifests
  if [[ -d "${RUN_ROOT}/policies/${policy}" ]]; then
    mkdir -p "${STAGE}/${RUN_ROOT}/policies"
    cp -r "${RUN_ROOT}/policies/${policy}" "${STAGE}/${RUN_ROOT}/policies/${policy}"
    # Drop heavyweight checkpoints if present
    find "${STAGE}/${RUN_ROOT}/policies/${policy}" -type d -name "global_step_*" -prune -exec rm -rf {} +
    find "${STAGE}/${RUN_ROOT}/policies/${policy}" -type d -name "actor" -prune -exec rm -rf {} +
    find "${STAGE}/${RUN_ROOT}/policies/${policy}" -type d -name "optimizer" -prune -exec rm -rf {} +
    find "${STAGE}/${RUN_ROOT}/policies/${policy}" -type d -name "hf_model" -prune -exec rm -rf {} +
    find "${STAGE}/${RUN_ROOT}/policies/${policy}" -type d -name "tensorboard_log" -prune -exec rm -rf {} +
  fi

  # selector pool for this policy
  if [[ -d "${RUN_ROOT}/selector_pools/${policy}" ]]; then
    mkdir -p "${STAGE}/${RUN_ROOT}/selector_pools"
    cp -r "${RUN_ROOT}/selector_pools/${policy}" "${STAGE}/${RUN_ROOT}/selector_pools/${policy}"
  fi

  # rollout snapshot files for this policy if available
  if [[ -d "${RUN_ROOT}/analysis/rollout_snapshots/${policy}" ]]; then
    mkdir -p "${STAGE}/${RUN_ROOT}/analysis/rollout_snapshots"
    cp -r "${RUN_ROOT}/analysis/rollout_snapshots/${policy}" "${STAGE}/${RUN_ROOT}/analysis/rollout_snapshots/${policy}"
  fi

  (
    cd "${STAGE}"
    zip -r "${OLDPWD}/${OUT_DIR}/${policy}_trace.zip" . >/dev/null
  )
done

echo "Wrote archives in: ${OUT_DIR}"
ls -lh "${OUT_DIR}"/*.zip
