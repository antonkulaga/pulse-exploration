#!/bin/bash
# E0 (no training): format stress test (E0a), GalaxyPPG watch vs E4 at rest (E0b), polarity detector (E0c).
#   1. one CPU job builds the E0 windows from raw VitalDB tracks and GalaxyPPG, and runs E0c;
#   2. every encoder embeds them: the foundation models on the second GPU pool (in parallel), the KronLSTM
#      encoders on an A100 through lib.sh's lock;
#   3. one CPU job scores them with the r0 probes (fitted on clean train embeddings, not refitted) and
#      writes summary.md.
# Paths come from slurm/env.local.sh; FM_GALAXYPPG is GalaxyPPG's unpacked Dataset folder.
#   setsid nohup fm_baselines/slurm/run_e0.sh > <log> 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
KRON="$KRON_FM"
R0="${FM_R0:-$FM_HOME/runs/r0-partial}"
OUT="${FM_E0_OUT:-$FM_HOME/runs/stress}"
GALAXY="${FM_GALAXYPPG:-$FM_HOME/galaxyppg/Dataset}"
V="$KRON/data/interim/ppg/vitaldb-partial/fs25-w30s"
PLETH="$KRON/data/input/ppg/vitaldb/pleth"
R="$KRON/data/output/runs/kron-ppg"
CASES="$KRON/data/input/ppg/vitaldb/cases.csv.gz"
BIDMC="${FM_EXT:-$FM_HOME/smoke}/bidmc/fs25-w30s"
W="$OUT/windows"
mkdir -p "$OUT"

log "start E0; out=$OUT"
if [ ! -f "$W/build.json" ] || [ ! -f "$W/build-galaxy.json" ]; then
  FM_CPU_CPUS=8 FM_CPU_MEM=48G cpu_job e0-build bash -c "python -m fm_baselines.e0 build-stress $V --pleth $PLETH --out $W --workers 8 && \
    python -m fm_baselines.e0 build-galaxy $GALAXY --out $W --workers 8 && \
    python -m fm_baselines.e0 polarity --stress $W --galaxy $W --bidmc $BIDMC --out $OUT/results" || { log "E0 build failed"; exit 1; }
fi
VARIANTS=$(ls -d "$W"/*/ | sed 's:/$::' | tr '\n' ' ')
pids=()
for m in pulseppg papagei_s anyppg; do
  small_gpu_job "e0-emb-$m" fm-baselines embed "$m" $VARIANTS --per-subject 0 --out "$OUT/emb" & pids+=($!)
  sleep 10
done
FM_RUN_IN="$KRON" gpu_job "e0-emb-kronlstm" bash -c "python -m fm_baselines.kron_embed $R/r1-pretrain/checkpoint.pt kronlstm_pretrain $VARIANTS --per-subject 0 --out $OUT/emb && \
  python -m fm_baselines.kron_embed $R/r2a-traits-finetune/checkpoint.pt kronlstm_traits_finetune $VARIANTS --per-subject 0 --out $OUT/emb" \
  || log "KronLSTM E0 embeddings failed"
for p in "${pids[@]}"; do wait "$p"; done
FM_CPU_CPUS=8 FM_CPU_MEM=48G cpu_job e0-score bash -c "python -m fm_baselines.e0 score --clean $R0/emb --emb $OUT/emb --cases $CASES --out $OUT/results && \
  cp $OUT/results/summary.md $OUT/summary.md"
log "== E0 finished"
cat "$OUT/summary.md" 2>/dev/null
