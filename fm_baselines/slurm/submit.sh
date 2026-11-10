#!/bin/bash
# fm_baselines/slurm/submit.sh gpu|cpu NAME COMMAND...   (NODE=gpuNN pins a GPU node)
# Cluster names (partition, account, node features) live in the git-ignored slurm/env.local.sh:
#   FM_GPU_PARTITION FM_CPU_PARTITION FM_ACCOUNT FM_GPU_CONSTRAINT FM_CPU_EXCLUDE UV_CACHE_DIR UV_PYTHON_INSTALL_DIR,
#   and for a second GPU pool FM_SMALL_GPU_NODES_ALL (its node names) FM_SMALL_GPU_PARTITION FM_SMALL_GPU_ACCOUNT
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.local.sh" ] && source "$HERE/env.local.sh"
PROJECT="$(cd "$HERE/.." && pwd)"
LOG_DIR="${LOG_DIR:-$PROJECT/slurm/logs}"
kind="${1:?gpu or cpu}"; name="${2:?job name}"; shift 2
[ $# -gt 0 ] || { echo "no command given" >&2; exit 2; }
mkdir -p "$LOG_DIR"
flags=(--job-name="fm-$name" --nodes=1 --ntasks=1 --output="$LOG_DIR/%x-%j.out")
[ -n "${FM_ACCOUNT:-}" ] && flags+=(--account="$FM_ACCOUNT")
if [ "$kind" = gpu ] && [ -n "${NODE:-}" ] && [[ " ${FM_SMALL_GPU_NODES_ALL:-} " == *" $NODE "* ]]; then
  # a node of the second GPU pool: its own partition, account and no node-feature constraint
  flags=(--job-name="fm-$name" --nodes=1 --ntasks=1 --output="$LOG_DIR/%x-%j.out"
         --account="${FM_SMALL_GPU_ACCOUNT:?set FM_SMALL_GPU_ACCOUNT in slurm/env.local.sh}"
         --partition="${FM_SMALL_GPU_PARTITION:?set FM_SMALL_GPU_PARTITION in slurm/env.local.sh}"
         --cpus-per-task="${FM_GPU_CPUS:-8}" --mem="${FM_SMALL_GPU_MEM:-64G}" --nodelist="$NODE")
elif [ "$kind" = gpu ]; then
  flags+=(--partition="${FM_GPU_PARTITION:?set FM_GPU_PARTITION in slurm/env.local.sh}"
          --cpus-per-task="${FM_GPU_CPUS:-8}" --mem="${FM_GPU_MEM:-160G}")
  [ -n "${FM_GPU_CONSTRAINT:-}" ] && flags+=(--constraint="$FM_GPU_CONSTRAINT")
  [ -n "${NODE:-}" ] && flags+=(--nodelist="$NODE")
else
  flags+=(--partition="${FM_CPU_PARTITION:?set FM_CPU_PARTITION in slurm/env.local.sh}"
          --cpus-per-task="${FM_CPU_CPUS:-8}" --mem="${FM_CPU_MEM:-64G}")
  [ -n "${FM_CPU_EXCLUDE:-}" ] && flags+=(--exclude="$FM_CPU_EXCLUDE")
fi
sbatch "${flags[@]}" --export="ALL,FM_PROJECT=$PROJECT,FM_JOB_KIND=$kind" "$HERE/job.sbatch" "$@"
