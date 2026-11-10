#!/bin/bash
# Overnight experiments on the r0 data (kron-ppg's 3,986-person vitaldb-partial windows):
#   1. format stress tests (stress.py): the three foundation models on the 2080 Ti pool, run 1's KronLSTM
#      on an A100 (its kernels need one), through lib.sh's A100 lock shared with run_all.sh;
#   2. the three foundation models fine-tuned on VitalDB (finetune.py), in parallel on the 2080 Ti pool: the
#      fair comparison with kron-ppg's encoder, which saw VitalDB in pretraining;
#   3. one CPU job: the r0 summary again, now with the fine-tuned models, and the stress summary.
#   setsid nohup fm_baselines/slurm/run_overnight.sh > <log> 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
KRON="$KRON_FM"
R0="${FM_R0:-$FM_HOME/runs/r0-partial}"
EXT="${FM_EXT:-$FM_HOME/smoke}"
V="$KRON/data/interim/ppg/vitaldb-partial/fs25-w30s"
R="$KRON/data/output/runs/kron-ppg"
CASES="$KRON/data/input/ppg/vitaldb/cases.csv.gz"
STEPS="${FM_FT_STEPS:-6000}"

log "start overnight; r0=$R0 steps=$STEPS"
pids=()
for m in pulseppg papagei_s anyppg; do
  ( small_gpu_job "stress-$m" python -m fm_baselines.stress "$m" --variant "$V" --clean-emb "$R0/emb/$m" \
      --cases "$CASES" --out "$R0/stress" || log "stress $m failed"
    small_gpu_job "ft-$m" python -m fm_baselines.finetune "$m" --variant "$V" --cases "$CASES" --steps "$STEPS" \
      --external "$EXT/bidmc/fs25-w30s" "$EXT/galaxyppg/fs25-w30s" \
      --out "$R0/probe/${m}_finetune/vitaldb-fs25-w30s" || log "finetune $m failed" ) &
  pids+=($!)
  sleep 20
done
FM_RUN_IN="$KRON" gpu_job "stress-kronlstm" python -m fm_baselines.stress kronlstm_pretrain \
  --kron "$R/r1-pretrain/checkpoint.pt" --variant "$V" --clean-emb "$R0/emb/kronlstm_pretrain" \
  --cases "$CASES" --out "$R0/stress" || log "stress kronlstm failed"
for p in "${pids[@]}"; do wait "$p"; done
cpu_job overnight-summary bash -c "fm-baselines summary $R0/probe > $R0/probe/summary.md && \
  python -m fm_baselines.stress x --variant $V --clean-emb . --out $R0/stress --summary > $R0/stress/summary.md"
log "== overnight finished"
cat "$R0/probe/summary.md" "$R0/stress/summary.md" 2>/dev/null
