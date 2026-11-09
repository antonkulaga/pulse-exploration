#!/bin/bash
# One fm-baselines command as a SLURM job: `uv run --no-sync <command...>` in the project, or, with
# FM_RUN_IN=<another uv project> (kron-fm, for the KronLSTM encoders), in that project's environment with
# this project's src on PYTHONPATH.
# The GPU nodes have no GRES, so a GPU job takes the first card with no compute process and exits 3
# when every card on its node is busy (the driver then tries another node).
set -euo pipefail
HERE="${FM_PROJECT:?}/slurm"
[ -f "$HERE/env.local.sh" ] && source "$HERE/env.local.sh"
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
cd "${FM_RUN_IN:-$FM_PROJECT}"
[ -n "${FM_RUN_IN:-}" ] && export PYTHONPATH="$FM_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} kind=${FM_JOB_KIND:-cpu} dir=$(pwd) commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
if [ "${FM_JOB_KIND:-cpu}" = gpu ]; then
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  dev=""
  while IFS=, read -r idx uuid; do
    uuid=$(echo "$uuid" | xargs)
    if ! grep -q "$uuid" <<<"$busy"; then dev=$(echo "$idx" | xargs); break; fi
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  [ -z "$dev" ] && { echo "every GPU on $(hostname) is busy" >&2; exit 3; }
  export CUDA_VISIBLE_DEVICES="$dev"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$dev"
else
  export CUDA_VISIBLE_DEVICES=""
fi
exec uv run --no-sync "$@"
