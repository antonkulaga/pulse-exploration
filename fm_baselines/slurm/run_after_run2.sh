#!/bin/bash
# Embeds kron-ppg run 2's prepared windows with each open encoder (one GPU job per model and rate) and
# probes them (one CPU job per rate), as soon as the windows exist:
#   * 25 Hz (vitaldb, bidmc, galaxyppg) after run 2's prepare job (named `prep=` in ppg-run2b.log);
#   * 125 Hz (vitaldb, bidmc) after the publish job's prepare (named `hub=` in ppg-publish.log).
# Run 2 pins gpu13 and its jobs fail when both cards there are taken, so this driver keeps off gpu13
# until run 2 has finished; a job that finds every card busy (exit 3) is resubmitted on the next node.
# Start it detached:  setsid nohup fm_baselines/slurm/run_after_run2.sh > <log> 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KRON="${KRON_FM:-/data/uq2347/sources/kron-fm}"
RUN2_LOG="${RUN2_LOG:-/data/uq2347/ppg-run2b.log}"
PUB_LOG="${PUB_LOG:-/data/uq2347/ppg-publish.log}"
OUT="${FM_OUT:-/data/uq2347/fm-baselines/runs/r1}"
MODELS="${FM_MODELS:-pulseppg papagei_s anyppg}"
I="$KRON/data/interim/ppg"
mkdir -p "$OUT"

log() { echo "$(date -u '+%F %H:%M') $*"; }
jid() { awk '/Submitted batch job/ {print $4}'; }
state() { sacct -j "$1" -X -n --format=State,ExitCode | head -1 | xargs; }
# 0 completed, 3 "every GPU busy", 1 any other end
wait_job() {
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
wait_line() {  # wait until a driver log names a job, then for that job
  local file=$1 key=$2 job
  until grep -q "^$key=" "$file" 2>/dev/null; do
    grep -qE "incomplete|ended" "$file" 2>/dev/null && { log "$file reports a failure before $key"; return 1; }
    sleep 60
  done
  job=$(awk -F= -v k="$key" '$1 == k {print $2}' "$file" | tail -1)
  log "waiting for $key job $job"
  wait_job "$job" || { log "$key job $job did not complete"; return 1; }
}
gpu_nodes() { if grep -q "run2 finished" "$RUN2_LOG" 2>/dev/null; then echo "gpu11 gpu12 gpu13"; else echo "gpu11 gpu12"; fi; }
gpu_job() {  # gpu_job NAME command...: submit, and move to another node while every card is busy
  local name=$1; shift
  local tries=0 node job rc
  while :; do
    for node in $(gpu_nodes); do
      job=$(NODE=$node "$HERE/submit.sh" gpu "$name" "$@" | jid)
      log "job $name $job on $node"
      wait_job "$job"; rc=$?
      [ $rc -eq 0 ] && { log "job $name $job completed"; return 0; }
      [ $rc -ne 3 ] && return 1
    done
    tries=$((tries + 1))
    [ $tries -ge 72 ] && { log "gave up on $name: no free GPU for 12 h"; return 1; }
    sleep 600
  done
}
cpu_job() {
  local name=$1; shift
  local job
  job=$("$HERE/submit.sh" cpu "$name" "$@" | jid)
  log "job $name $job on cpu"
  wait_job "$job" && log "job $name $job completed"
}
phase() {  # phase RATE variant-dirs...
  local rate=$1; shift
  local pids=() m ok=0
  for m in $MODELS; do
    gpu_job "emb$rate-$m" fm-baselines embed "$m" "$@" --out "$OUT/emb" & pids+=($!)
    sleep 5
  done
  for p in "${pids[@]}"; do wait "$p" || ok=1; done
  cpu_job "probe$rate" fm-baselines probe "$OUT/emb" --out "$OUT/probe" || ok=1
  return $ok
}

log "start; out=$OUT models=$MODELS"
if wait_line "$RUN2_LOG" prep; then
  phase 25 "$I/vitaldb/fs25-w30s" "$I/bidmc/fs25-w30s" "$I/galaxyppg/fs25-w30s" || log "25 Hz phase had failures"
fi
if wait_line "$PUB_LOG" hub; then
  phase 125 "$I/vitaldb/fs125-w30s" "$I/bidmc/fs125-w30s" || log "125 Hz phase had failures"
fi
log "== fm-baselines finished"
cat "$OUT/probe/summary.md" 2>/dev/null
