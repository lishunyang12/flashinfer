#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlashInfer contributors.

set -euo pipefail

devices="${1:-0,1,2,3,4,5,6,7}"
command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi not found" >&2
  exit 2
}
command -v numactl >/dev/null || {
  echo "ERROR: numactl not found" >&2
  exit 2
}

IFS=',' read -r -a gpu_list <<<"${devices}"
declare -A seen=()
available="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')"
for gpu in "${gpu_list[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid GPU index ${gpu}" >&2
    exit 2
  }
  [[ -z "${seen[${gpu}]:-}" ]] || {
    echo "ERROR: duplicate GPU index ${gpu}" >&2
    exit 2
  }
  grep -qx "${gpu}" <<<"${available}" || {
    echo "ERROR: GPU index ${gpu} does not exist" >&2
    exit 2
  }
  seen[${gpu}]=1
done

echo "# selected GPUs: ${devices}"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader | awk -F', ' -v list="${devices}" '
  BEGIN {split(list, a, ","); for (i in a) wanted[a[i]]=1}
  wanted[$1] {print}
'

busy=0
for gpu in "${gpu_list[@]}"; do
  uuid="$({
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
      awk -F', ' -v i="${gpu}" '$1 == i {print $2}'
  })"
  count="$({
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null |
      grep -c "${uuid}" || true
  })"
  busy=$((busy + count))
done

echo "# topology"
nvidia-smi topo -m
echo "# NUMA"
numactl --hardware

if ((busy != 0)); then
  echo "REFUSED: selected GPUs have ${busy} active compute process(es)" >&2
  exit 3
fi
echo "PREFLIGHT: PASS (selected GPUs idle)"
