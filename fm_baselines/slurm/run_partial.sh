#!/bin/bash
# Preliminary run (r0) on what existed before run 2: kron-ppg's 3,986-person `vitaldb-partial` windows,
# run 1's KronLSTM encoder, the r2a trait heads and the r1 clocks; run 1's BIDMC and GalaxyPPG windows
# converted to parquet as the external sets. Two GPU jobs (foundation models; KronLSTM encoders), then one
# CPU job for the probes, kron-ppg's own heads and summary.md.
#   NODE=<a free A100 node> setsid nohup fm_baselines/slurm/run_partial.sh > <log> 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.local.sh" ] && source "$HERE/env.local.sh"
FM_HOME="${FM_HOME:-$HOME/fm-baselines}"
KRON="${KRON_FM:-$HOME/sources/kron-fm}"
OUT="${FM_OUT:-$FM_HOME/runs/r0-partial}"
EXT="${FM_EXT:-$FM_HOME/smoke}"  # bidmc/ and galaxyppg/ fs25-w30s in the parquet format
I="$KRON/data/interim/ppg"
R="$KRON/data/output/runs/kron-ppg"
CASES="$KRON/data/input/ppg/vitaldb/cases.csv.gz"
V="$I/vitaldb-partial/fs25-w30s $EXT/bidmc/fs25-w30s $EXT/galaxyppg/fs25-w30s"
log() { echo "$(date -u '+%F %H:%M') $*"; }
jid() { awk '/Submitted batch job/ {print $4}'; }
wait_job() {
  local s
  while :; do
    s=$(sacct -j "$1" -X -n --format=State,ExitCode | head -1 | xargs)
    case "$s" in
      COMPLETED*) log "job $1 completed"; return 0 ;;
      PENDING*|RUNNING*|REQUEUED*|CONFIGURING*|COMPLETING*|"") sleep 30 ;;
      *) log "job $1 ended $s"; return 1 ;;
    esac
  done
}
fm=$("$HERE/submit.sh" gpu r0-emb-fm bash -c \
  "for m in pulseppg papagei_s anyppg; do fm-baselines embed \$m $V --out $OUT/emb || exit 1; done" | jid)
log "job r0-emb-fm $fm"
sleep 20
kron=$(FM_RUN_IN="$KRON" "$HERE/submit.sh" gpu r0-emb-kron bash -c \
  "python -m fm_baselines.kron_embed $R/r1-pretrain/checkpoint.pt kronlstm_pretrain $V --out $OUT/emb && \
   python -m fm_baselines.kron_embed $R/r1-pretrain/checkpoint.pt kronlstm_untrained $V --out $OUT/emb --random && \
   python -m fm_baselines.kron_embed $R/r2a-traits-finetune/checkpoint.pt kronlstm_traits_finetune $V --out $OUT/emb" | jid)
log "job r0-emb-kron $kron"
wait_job "$fm"; a=$?
wait_job "$kron"; b=$?
probe=$("$HERE/submit.sh" cpu r0-probe bash -c \
  "fm-baselines probe $OUT/emb --cases $CASES --out $OUT/probe && \
   fm-baselines kron-eval $R/r2a-traits-probe $R/r2a-traits-finetune $R/r1-clock-scratch $R/r1-clock-finetune \
     $R/r1-clock-probe --subjects $I/vitaldb-partial/fs25-w30s/subjects.parquet --cases $CASES --out $OUT/probe" | jid)
log "job r0-probe $probe (embeddings: fm $a, kron $b)"
wait_job "$probe"
log "== r0 finished"
cat "$OUT/probe/summary.md" 2>/dev/null
