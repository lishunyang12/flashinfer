#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlashInfer contributors.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
repo_parent="$(dirname "${repo_root}")"
gpu_id="${GPU_ID:-0}"
base_python="${PYTHON_BIN:-${repo_parent}/minimax-h3-native/.venv/bin/python}"
validation_overlay="${VALIDATION_OVERLAY:-${repo_parent}/.flashinfer-sm120-validation}"
export FLASHINFER_WORKSPACE_BASE="${JIT_CACHE_BASE:-${repo_parent}/.flashinfer-sm120-jit}"
result_base="${RESULT_BASE:-${repo_parent}/sm120-skip-benchmark-results}"
result_root="${result_base}/$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${result_root}" "${FLASHINFER_WORKSPACE_BASE}"
exec > >(tee "${result_root}/benchmark.log") 2>&1

echo "RESULT_ROOT=${result_root}"
echo "REPO_ROOT=${repo_root}"
echo "GPU_ID=${gpu_id}"
echo "PYTHON_BIN=${base_python}"
echo "FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"
echo "FLASHINFER_COMMIT=$(git -C "${repo_root}" rev-parse HEAD)"
hostname
date -u

if [[ ! -x "${base_python}" ]]; then
  echo "ERROR: Python not found at ${base_python}; set PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "${validation_overlay}" ]]; then
  echo "ERROR: validation overlay not found; run validate_sm120_prims_skip_softmax.sh first" >&2
  exit 2
fi

gpu_uuid="$(nvidia-smi -i "${gpu_id}" --query-gpu=uuid --format=csv,noheader)"
busy_count="$(
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
    | grep -c "${gpu_uuid}" || true
)"
if ((busy_count != 0)); then
  echo "REFUSED: GPU ${gpu_id} has ${busy_count} active compute process(es)" >&2
  exit 3
fi
nvidia-smi -i "${gpu_id}" \
  --query-gpu=index,name,compute_cap,driver_version,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

validation_pythonpath="${repo_root}:${validation_overlay}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${validation_pythonpath}" \
  "${base_python}" "${repo_root}/benchmarks/bench_sm120_skip_softmax.py" "$@"
