#!/bin/bash
# The E0 battery for new kron-ppg encoders: B1 (format stress, E0a), B3 (GalaxyPPG watch vs E4 at rest) and
# B4 (GalaxyPPG at rest vs VitalDB people under 30), all from E0b/E0a's windows (run_e0.sh builds them).
# kron-ppg lists its encoders in a brief (KRON_BATTERY_BRIEF); a run is picked up when its name appears there
# and its folder holds checkpoint.pt and metrics.json (training finished). For each run, one A100 job embeds
# the r0 VitalDB windows (to fit the probes) and the E0 windows; one CPU job scores it. Polls every 10 min
# for up to FM_BATTERY_HOURS (default 14).
#   setsid nohup fm_baselines/slurm/run_battery.sh > <log> 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
KRON="$KRON_FM"
R="$KRON/data/output/runs/kron-ppg"
BRIEF="${KRON_BATTERY_BRIEF:-$KRON_LOGS/briefs/kron-ppg-runs-for-battery.md}"
W="${FM_E0_OUT:-$FM_HOME/runs/stress}/windows"
OUT="${FM_BATTERY_OUT:-$FM_HOME/runs/battery}"
V="$KRON/data/interim/ppg/vitaldb-partial/fs25-w30s"
CASES="$KRON/data/input/ppg/vitaldb/cases.csv.gz"
HOURS="${FM_BATTERY_HOURS:-14}"
mkdir -p "$OUT"
touch "$OUT/done.txt"

rate_of() {  # the sample rate a kron-ppg checkpoint reads
  (cd "$KRON" && uv run --no-sync python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['config']['sample_rate'])" "$1" 2>/dev/null)
}
score_run() {
  local run=$1 ckpt="$R/$1/checkpoint.pt" rate
  rate=$(rate_of "$ckpt")
  if [ "${rate%.*}" != "25" ]; then
    log "skip $run: its encoder reads ${rate:-?} Hz and the E0 windows are 25 Hz"; echo "$run skipped rate=$rate" >> "$OUT/done.txt"; return
  fi
  local variants clean_v="$V" vname
  variants=$(ls -d "$W"/*/ | sed 's:/$::' | tr '\n' ' ')
  # a run trained on its own variant (the brief names it, e.g. fs25-w30s-canon) gets its probes fitted on that
  # variant's VitalDB windows; the E0 windows keep the standard preparation, which is what a canon front end
  # has to be robust to
  vname=$(grep -w -- "$run" "$BRIEF" | grep -oE "fs25-w30s[-a-z0-9]*" | head -1)
  if [ -n "$vname" ] && [ "$vname" != "fs25-w30s" ] && [ -f "$KRON/data/interim/ppg/vitaldb/$vname/windows.parquet" ]; then
    clean_v="$KRON/data/interim/ppg/vitaldb/$vname"
    log "$run: probes fitted on VitalDB $vname"
  fi
  FM_RUN_IN="$KRON" gpu_job "battery-emb-$run" bash -c "python -m fm_baselines.kron_embed $ckpt $run $clean_v --out $OUT/clean && \
    python -m fm_baselines.kron_embed $ckpt $run $variants --per-subject 0 --out $OUT/emb" || { log "battery embeddings for $run failed"; return; }
  if [ "$clean_v" != "$V" ]; then  # the scorer reads the probe's training set as vitaldb-fs25-w30s.*
    for f in "$OUT/clean/$run/vitaldb-$vname".*; do mv "$f" "${f/vitaldb-$vname/vitaldb-fs25-w30s}"; done
  fi
  FM_CPU_CPUS=4 FM_CPU_MEM=32G cpu_job "battery-score-$run" python -m fm_baselines.e0 score --clean "$OUT/clean" --emb "$OUT/emb" \
    --encoders "$run" --cases "$CASES" --out "$OUT/results" && echo "$run" >> "$OUT/done.txt"
  (cd "$HERE/.." && uv run --no-sync python -m fm_baselines.e0 summary "$OUT/results" > "$OUT/summary.md")
}

log "start battery; brief=$BRIEF out=$OUT"
until [ -f "$W/build.json" ] && [ -f "$W/build-galaxy.json" ]; do sleep 120; done
end=$(( $(date +%s) + HOURS * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  if [ -f "$BRIEF" ]; then
    for d in "$R"/*/; do
      run=$(basename "$d")
      grep -qw -- "$run" "$BRIEF" || continue
      grep -qx -- "$run" "$OUT/done.txt" && continue
      grep -q -- "^$run skipped" "$OUT/done.txt" && continue
      [ -f "$d/checkpoint.pt" ] && [ -f "$d/metrics.json" ] || continue
      log "scoring $run"
      score_run "$run"
    done
  fi
  sleep 600
done
log "== battery stopped after $HOURS h"
