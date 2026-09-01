#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlashInfer contributors.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
repo_parent="$(dirname "${repo_root}")"
gpu_id="${GPU_ID:-4}"
numa_node=$((gpu_id / 4))
pr_ref="origin/sm120-prims-skip-softmax-pr"
result_root="${RESULT_ROOT:-${repo_parent}/sm120-skip-fraction-sweep/$(date -u +%Y%m%dT%H%M%SZ)}"
fractions=(0 0.25 0.5 0.75 0.9 0.99 0.998)

mkdir -p "${result_root}"
exec > >(tee "${result_root}/sweep.log") 2>&1

echo "RESULT_ROOT=${result_root}"
echo "PERF_COMMIT=$(git -C "${repo_root}" rev-parse HEAD)"
echo "PR_COMMIT=$(git -C "${repo_root}" rev-parse "${pr_ref}")"
echo "GPU_ID=${gpu_id}"
echo "NUMA_NODE=${numa_node}"

runtime_files=(
  flashinfer/attention/cute_dsl/sm120_fmha.py
  flashinfer/cute_dsl/attention/fmha/sm120/compile.py
  flashinfer/cute_dsl/attention/fmha/sm120/fmha_prefill_fp8_tma.py
  flashinfer/prefill.py
)
if ! git -C "${repo_root}" diff --quiet "${pr_ref}" HEAD -- "${runtime_files[@]}"; then
  echo "ERROR: performance branch runtime differs from ${pr_ref}" >&2
  exit 2
fi

bash "${repo_root}/scripts/pro5000_preflight.sh" "${gpu_id}"

for fraction in "${fractions[@]}"; do
  echo "SWEEP_START active_skip_fraction=${fraction}"
  numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
    env RESULT_BASE="${result_root}/fraction-${fraction}" GPU_ID="${gpu_id}" \
    bash "${repo_root}/scripts/benchmark_sm120_skip_softmax.sh" \
      --preset custom \
      --q-len 63744 \
      --kv-len 63744 \
      --num-q-heads 56 \
      --num-kv-heads 56 \
      --head-dim 128 \
      --skip-scale-factor 1.0 \
      --active-skip-fraction "${fraction}" \
      --warmup 5 \
      --repeat 30
done

echo "SWEEP_ROOT=${result_root}"
grep -hE '^case=|^mode=|^CSV=|SM120_SKIP_SOFTMAX_BENCHMARK:' \
  "${result_root}"/fraction-*/*/benchmark.log
