#!/usr/bin/env bash
set -euo pipefail

# Package policy-grid traces for remote debugging/analysis.
# Excludes heavyweight artifacts (envs, checkpoints, model weights).
#
# Usage:
#   bash scripts/package_policy_grid_trace.sh runs/policy_grid_8gpu
#   bash scripts/package_policy_grid_trace.sh runs/policy_grid_8gpu /tmp/policy_grid_trace.zip

RUN_ROOT="${1:-runs/policy_grid_8gpu}"
OUT_ZIP="${2:-${RUN_ROOT%/}_trace_$(date +%Y%m%d_%H%M%S).zip}"

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Run root not found: ${RUN_ROOT}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

STAGE="${TMP_DIR}/payload"
mkdir -p "${STAGE}"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -r "${src}" "${dst}"
  fi
}

# Top-level manifests and key logs
copy_if_exists "${RUN_ROOT}/policy_grid_manifest.json" "${STAGE}/${RUN_ROOT}/policy_grid_manifest.json"
copy_if_exists "${RUN_ROOT}/profile.log" "${STAGE}/${RUN_ROOT}/profile.log"
copy_if_exists "${RUN_ROOT}/progress" "${STAGE}/${RUN_ROOT}/progress"
copy_if_exists "${RUN_ROOT}/analysis" "${STAGE}/${RUN_ROOT}/analysis"
copy_if_exists "${RUN_ROOT}/profile" "${STAGE}/${RUN_ROOT}/profile"
copy_if_exists "${RUN_ROOT}/selector_pools" "${STAGE}/${RUN_ROOT}/selector_pools"

# Per-policy logs/manifests (without checkpoints)
while IFS= read -r -d '' f; do
  rel="${f#./}"
  dst="${STAGE}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  cp "${f}" "${dst}"
done < <(
  find . -type f \
    \( -path "./${RUN_ROOT}/policies/*/launcher.log" \
    -o -path "./${RUN_ROOT}/policies/*/branch.log" \
    -o -path "./${RUN_ROOT}/policies/*/periodic_gradient_selector_manifest.json" \
    -o -path "./${RUN_ROOT}/policies/*/periodic/*.log" \
    -o -path "./${RUN_ROOT}/policies/*/periodic/**/*.log" \
    -o -path "./${RUN_ROOT}/policies/*/periodic/**/*.json" \
    -o -path "./${RUN_ROOT}/policies/*/periodic/**/*.csv" \) \
    -print0 2>/dev/null
)

# Helpful repo context for reproducing parsing/analysis
mkdir -p "${STAGE}/scripts"
for f in \
  scripts/track_policy_grid_progress.py \
  scripts/export_policy_learning_delta.py \
  scripts/plot_policy_grid_curves.py \
  scripts/show_policy_step_table.py \
  scripts/show_policy_curves_ascii.py \
  scripts/build_policy_grid_report.py \
  scripts/collect_policy_rollout_snapshots.py \
  scripts/report_newly_solved_examples.py \
  scripts/run_policy_grid_8gpu.py \
  scripts/run_periodic_gradient_selector.py \
  scripts/profile_grpo_ground_truth.py; do
  if [[ -f "${f}" ]]; then
    cp "${f}" "${STAGE}/${f}"
  fi
done

# Snapshot git metadata (if present)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  mkdir -p "${STAGE}/meta"
  git rev-parse HEAD > "${STAGE}/meta/git_head.txt" || true
  git status --short > "${STAGE}/meta/git_status_short.txt" || true
  git diff -- scripts > "${STAGE}/meta/git_diff_scripts.patch" || true
fi

# Build zip from staging dir
OUT_DIR="$(dirname "${OUT_ZIP}")"
mkdir -p "${OUT_DIR}"
(
  cd "${STAGE}"
  zip -r "${OLDPWD}/${OUT_ZIP}" . >/dev/null
)

echo "Wrote trace archive: ${OUT_ZIP}"
echo "Archive size:"
ls -lh "${OUT_ZIP}"
