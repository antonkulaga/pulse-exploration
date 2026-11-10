#!/bin/bash
# Shared functions for the fm_baselines drivers (source it). Two GPU pools:
#   * the A100 nodes (gpu_job): KronLSTM jobs only, through one lock, so fm_baselines holds at most one A100
#     card at a time; those cards are shared with kron-ppg, kron-dna and other users;
#   * the 2080 Ti nodes (small_gpu_job): the foundation models, on the preemptible partition (a requeued job
#     simply runs again); every job checks for a free card itself and exits 3 when there is none.
# Paths and node names come from the git-ignored slurm/env.local.sh (see submit.sh):
#   FM_HOME (work folder), KRON_FM (kron-fm checkout), KRON_LOGS (folder of kron-ppg's driver logs),
#   FM_A100_NODES, FM_KRON_GPU_NODE (the node kron-ppg pins), FM_SMALL_GPU_NODES.
FM_LIB_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$FM_LIB_HERE/env.local.sh" ] && source "$FM_LIB_HERE/env.local.sh"
FM_HOME="${FM_HOME:-$HOME/fm-baselines}"
KRON_FM="${KRON_FM:-$HOME/sources/kron-fm}"
KRON_LOGS="${KRON_LOGS:-$HOME}"
FM_LOCK="${FM_LOCK:-$FM_HOME/gpu.lock}"
TRAITS_LOG="${TRAITS_LOG:-$KRON_LOGS/ppg-traits-b.log}"

log() { echo "$(date -u '+%F %H:%M') $*"; }
jid() { awk '/Submitted batch job/ {print $4}'; }
state() { sacct -j "$1" -X -n --format=State,ExitCode | head -1 | xargs; }
wait_job() {  # 0 completed, 3 every GPU on the node busy, 1 any other end
  local s
  while :; do
    s=$(state "$1")
    case "$s" in
      COMPLETED*) return 0 ;;
      FAILED*" 3:0") return 3 ;;
      PENDING*|RUNNING*|REQUEUED*|CONFIGURING*|COMPLETING*|"") sleep 60 ;;
      *) log "job $1 ended $s"; return 1 ;;
    esac
  done
}
wait_key() {  # wait_key LOG KEY: wait until LOG names `KEY=<job>` (alone or among others), then for that job
  local file=$1 key=$2 job
  until job=$(grep -oE "(^| )$key=[0-9]+" "$file" 2>/dev/null | tail -1 | cut -d= -f2) && [ -n "$job" ]; do
    grep -qE "incomplete|no run 2 encoder" "$file" 2>/dev/null && { log "$file stopped before $key"; return 1; }
    sleep 60
  done
  log "waiting for $key job $job"
  wait_job "$job" || { log "$key job $job did not complete"; return 1; }
}
wait_text() {  # wait_text LOG TEXT
  until grep -q "$2" "$1" 2>/dev/null; do
    grep -qE "incomplete|no run 2 encoder" "$1" 2>/dev/null && { log "$1 stopped before '$2'"; return 1; }
    sleep 120
  done
}
gpu_lock() {  # mkdir is atomic on BeeGFS; a lock whose holder died is taken over
  while ! mkdir "$FM_LOCK" 2>/dev/null; do
    local holder
    holder=$(cat "$FM_LOCK/pid" 2>/dev/null)
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then rm -rf "$FM_LOCK"; continue; fi
    sleep 30
  done
  echo "${BASHPID:-$$}" > "$FM_LOCK/pid"
}
gpu_unlock() { rm -rf "$FM_LOCK"; }
gpu_nodes() {  # kron-ppg pins one node and its jobs fail when both cards there are taken: keep off it until done
  if grep -qE "traits finished|no run 2 encoder" "$TRAITS_LOG" 2>/dev/null; then echo "$FM_A100_NODES $FM_KRON_GPU_NODE"
  else echo "$FM_A100_NODES"; fi
}
gpu_job() {  # gpu_job NAME command...; FM_RUN_IN in the environment runs it in another uv project (kron-fm)
  local name=$1; shift
  local tries=0 node job rc
  gpu_lock
  while :; do
    for node in $(gpu_nodes); do
      job=$(NODE=$node "$FM_LIB_HERE/submit.sh" gpu "$name" "$@" | jid)
      log "job $name $job on $node"
      wait_job "$job"; rc=$?
      [ $rc -eq 0 ] && { log "job $name $job completed"; gpu_unlock; return 0; }
      [ $rc -ne 3 ] && { gpu_unlock; return 1; }
    done
    tries=$((tries + 1))
    [ $tries -ge 72 ] && { log "gave up on $name: no free GPU for 12 h"; gpu_unlock; return 1; }
    gpu_unlock; sleep 600; gpu_lock  # let other fm_baselines jobs try while we wait
  done
}
small_gpu_nodes() {  # the 2080 Ti nodes (11 GB): enough for the foundation models, not for the KronLSTM kernels
  echo "$FM_SMALL_GPU_NODES"
}
small_gpu_job() {  # small_gpu_job NAME command...: the next small-GPU node with a free card; no lock (36 cards)
  local name=$1; shift
  local tries=0 node job rc
  while :; do
    for node in $(small_gpu_nodes); do
      job=$(NODE=$node "$FM_LIB_HERE/submit.sh" gpu "$name" "$@" | jid)
      log "job $name $job on $node"
      wait_job "$job"; rc=$?
      [ $rc -eq 0 ] && { log "job $name $job completed"; return 0; }
      [ $rc -ne 3 ] && return 1
    done
    tries=$((tries + 1))
    [ $tries -ge 72 ] && { log "gave up on $name: no free small GPU for 12 h"; return 1; }
    sleep 600
  done
}
cpu_job() {  # gpu_or_cpu_job NAME command...: one try on each GPU node; when every card is busy, run the
  # command on a CPU node with `--device cpu` appended (for the small foundation models only)
  local name=$1; shift
  local node job rc
  gpu_lock
  for node in $(gpu_nodes); do
    job=$(NODE=$node "$FM_LIB_HERE/submit.sh" gpu "$name" "$@" | jid)
    log "job $name $job on $node"
    wait_job "$job"; rc=$?
    [ $rc -eq 0 ] && { log "job $name $job completed"; gpu_unlock; return 0; }
    [ $rc -ne 3 ] && { gpu_unlock; return 1; }
  done
  gpu_unlock
  log "no free GPU for $name: running it on CPU"
  FM_CPU_CPUS="${FM_CPU_CPUS:-16}" FM_CPU_MEM="${FM_CPU_MEM:-96G}" cpu_job "$name-cpu" "$@" --device cpu
}
cpu_job() {
  local name=$1; shift
  local job
  job=$("$FM_LIB_HERE/submit.sh" cpu "$name" "$@" | jid)
  log "job $name $job on cpu"
  wait_job "$job" && log "job $name $job completed"
}
