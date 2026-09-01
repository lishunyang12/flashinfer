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
export FLASHINFER_WORKSPACE_BASE="${JIT_CACHE_BASE:-${repo_parent}/.flashinfer-sm120-bf16-probe-jit}"
result_base="${RESULT_BASE:-${repo_parent}/sm120-bf16-probe-results}"
result_root="${result_base}/$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${result_root}" "${FLASHINFER_WORKSPACE_BASE}"
exec > >(tee "${result_root}/probe.log") 2>&1

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

bash "${repo_root}/scripts/pro5000_preflight.sh" "${gpu_id}"

validation_pythonpath="${repo_root}:${validation_overlay}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${validation_pythonpath}" \
  "${base_python}" - <<'PY'
from importlib.metadata import version

import torch

print("nvidia-cutlass-dsl=", version("nvidia-cutlass-dsl"))
print("apache-tvm-ffi=", version("apache-tvm-ffi"))
print("runtime_compute_capability=", torch.cuda.get_device_capability(0))
PY

CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${validation_pythonpath}" \
  "${base_python}" "${repo_root}/benchmarks/probe_sm120_bf16_skip_softmax.py" "$@"
