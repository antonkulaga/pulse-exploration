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
# GPU jobs go through lib.sh: one card at a time for all fm_baselines drivers, off kron-ppg's node until its
# trait jobs have finished, and a job that finds every card busy (exit 3) moves to the next node.
# Start detached:  setsid nohup fm_baselines/slurm/run_all.sh > <log> 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
KRON="$KRON_FM"
LOGS="$KRON_LOGS"
RUN2_LOG="$LOGS/ppg-run2b.log" PUB_LOG="$LOGS/ppg-publish.log"
export TRAITS_LOG="$LOGS/ppg-traits-b.log"
OUT="${FM_OUT:-$FM_HOME/runs/r1}"
MODELS="${FM_MODELS:-pulseppg papagei_s anyppg}"
I="$KRON/data/interim/ppg"
R="$KRON/data/output/runs/kron-ppg"
CASES="$KRON/data/input/ppg/vitaldb/cases.csv.gz"
mkdir -p "$OUT"

fm_embed() {  # fm_embed RATE variant-dirs...
  local rate=$1; shift
  local m ok=0
  for m in $MODELS; do  # one after another, on the 2080 Ti pool (lib.sh)
    small_gpu_job "emb$rate-$m" fm-baselines embed "$m" "$@" --out "$OUT/emb" || ok=1
  done
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
