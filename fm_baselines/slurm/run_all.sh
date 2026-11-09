#!/bin/bash
# KronLSTM (kron-ppg) and open PPG foundation models on the same people, traits and evaluation, chained to
# kron-ppg's own drivers on the cluster. Every step starts as soon as its inputs exist:
#   A  25 Hz windows ready (`prep=` job in ppg-run2b.log)      -> Pulse-PPG, PaPaGei-S, AnyPPG embeddings (GPU)
#   B  run 2 pretraining done (`pretrain=` job in ppg-run2b.log) -> KronLSTM r2-pretrain encoder + an untrained
#                                                                   twin, embedded at 25 Hz (GPU)
#   C  kron-ppg trait heads done (`traits finished` in ppg-traits-b.log) -> the fine-tuned trait encoder (GPU)
#   D  125 Hz windows ready (`hub=` job in ppg-publish.log)      -> the three models at 125 Hz (GPU)
#   then one CPU job: probes for every embedding set, kron-ppg's own heads scored the same way, summary.md.
# An interim CPU probe runs after A and B, so the first numbers do not wait for C and D.
# kron-ppg pins gpu13 and its jobs fail when both cards there are taken, so this driver keeps off gpu13 until
# kron-ppg's trait jobs have finished; a job that finds every card busy (exit 3) moves to the next node.
# Start detached:  setsid nohup fm_baselines/slurm/run_all.sh > <log> 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KRON="${KRON_FM:-/data/uq2347/sources/kron-fm}"
LOGS="${KRON_LOGS:-/data/uq2347}"
RUN2_LOG="$LOGS/ppg-run2b.log" TRAITS_LOG="$LOGS/ppg-traits-b.log" PUB_LOG="$LOGS/ppg-publish.log"
OUT="${FM_OUT:-/data/uq2347/fm-baselines/runs/r1}"
MODELS="${FM_MODELS:-pulseppg papagei_s anyppg}"
I="$KRON/data/interim/ppg"
R="$KRON/data/output/runs/kron-ppg"
CASES="$KRON/data/input/ppg/vitaldb/cases.csv.gz"
mkdir -p "$OUT"

log() { echo "$(date -u '+%F %H:%M') $*"; }
jid() { awk '/Submitted batch job/ {print $4}'; }
state() { sacct -j "$1" -X -n --format=State,ExitCode | head -1 | xargs; }
wait_job() {  # 0 completed, 3 every GPU busy, 1 any other end
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
gpu_nodes() {
  if grep -qE "traits finished|no run 2 encoder" "$TRAITS_LOG" 2>/dev/null; then echo "gpu11 gpu12 gpu13"
  else echo "gpu11 gpu12"; fi
}
gpu_job() {  # gpu_job NAME command...; FM_RUN_IN in the environment runs it in another uv project
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
fm_embed() {  # fm_embed RATE variant-dirs...
  local rate=$1; shift
  local pids=() m ok=0
  for m in $MODELS; do
    gpu_job "emb$rate-$m" fm-baselines embed "$m" "$@" --out "$OUT/emb" & pids+=($!)
    sleep 5
  done
  for p in "${pids[@]}"; do wait "$p" || ok=1; done
  return $ok
}
kron_embed() {  # kron_embed NAME CHECKPOINT [--random]
  local name=$1 ckpt=$2; shift 2
  FM_RUN_IN="$KRON" gpu_job "emb25-$name" python -m fm_baselines.kron_embed "$ckpt" "$name" \
    "$I/vitaldb/fs25-w30s" "$I/bidmc/fs25-w30s" "$I/galaxyppg/fs25-w30s" --out "$OUT/emb" "$@"
}
probe_all() {  # probe_all NAME: every embedding set, then kron-ppg's own heads, one CPU job
  cpu_job "$1" bash -c "fm-baselines probe $OUT/emb --cases $CASES --out $OUT/probe && \
    runs=\$(ls -d $R/r2-traits-probe $R/r2-traits-finetune $R/r2-clock-scratch $R/r2-clock-finetune $R/r2-clock-probe 2>/dev/null); \
    if [ -n \"\$runs\" ]; then fm-baselines kron-eval \$runs --subjects $I/vitaldb/fs25-w30s/subjects.parquet \
      --cases $CASES --out $OUT/probe; fi"
}

log "start; out=$OUT models=$MODELS"
( wait_key "$RUN2_LOG" prep && fm_embed 25 "$I/vitaldb/fs25-w30s" "$I/bidmc/fs25-w30s" "$I/galaxyppg/fs25-w30s" ) &
A=$!
( wait_key "$RUN2_LOG" pretrain && kron_embed kronlstm_pretrain "$R/r2-pretrain/checkpoint.pt" \
    && kron_embed kronlstm_untrained "$R/r2-pretrain/checkpoint.pt" --random ) &
B=$!
( wait_text "$TRAITS_LOG" "traits finished" && [ -f "$R/r2-traits-finetune/checkpoint.pt" ] \
    && kron_embed kronlstm_traits_finetune "$R/r2-traits-finetune/checkpoint.pt" ) &
C=$!
( wait_key "$PUB_LOG" hub && fm_embed 125 "$I/vitaldb/fs125-w30s" "$I/bidmc/fs125-w30s" ) &
D=$!
wait $A; wait $B
probe_all probe-interim || log "interim probe failed"
wait $C; wait $D
probe_all probe-final || log "final probe failed"
log "== fm-baselines finished"
cat "$OUT/probe/summary.md" 2>/dev/null
