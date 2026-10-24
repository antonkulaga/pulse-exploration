"""False alerts of the beat-interval AF screen on GalaxyPPG, where nobody has AF.

GalaxyPPG's 23 watch participants are healthy adults aged 20-29 in sinus rhythm, so every irregular
segment is a false positive. Three beat sources are screened with the same rule
(`pulse_exploration.hrv.af_screen`, 64-beat segments, Lee et al. 2013 thresholds):
- ecg: chest-strap ECG R-peaks, NN intervals only (how often the rule itself fires on sinus rhythm)
- ppg25_hrv: foot_template beats from the watch's 25 Hz green PPG with its HRV settings
- ppg25_af: the same without interval-based artifact correction (`af_screen.AF_SETTINGS`), which the
  screen needs: on synthetic pulse waves the HRV cleaning discarded every AF segment
- samsung: Samsung's IBI list after the cleaning in `bench.SamsungIBIClean`
Sensitivity needs AF recordings, which GalaxyPPG does not have.

    uv run python benchmarks/af25/run.py [--out data/benchmarks/af25] [--workers N]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pulse_exploration import galaxyppg  # noqa: E402
from pulse_exploration.hrv import af_screen, bench  # noqa: E402
from pulse_exploration.hrv.af_screen import AF_SETTINGS  # noqa: E402
from pulse_exploration.hrv.cand_foot_template import FootTemplateEstimator  # noqa: E402


def participant(uid: str) -> list[dict]:
    rec = galaxyppg.load(uid)
    if not rec.has_watch or not rec.sessions:
        return []
    srcs: dict[str, list[af_screen.Segment]] = {}
    ref = bench._prepare_ref(uid)
    t = ref.t
    srcs["ecg"] = af_screen.segments_from_intervals(t[1:], np.diff(t), ref.nn, np.ones(len(t) - 2, bool))
    pt, pv = rec.ppg_arrays()
    at, axyz = rec.acc_arrays()
    for name, kw in (("ppg25_hrv", {}), ("ppg25_af", AF_SETTINGS)):
        out = FootTemplateEstimator(**kw).estimate(pt, pv, at, axyz)
        srcs[name] = af_screen.segments_from_beats(out["beats_ms"], np.asarray(out["quality"]) >= 0.5)
    s = bench.SamsungIBIClean().estimate_recording(rec)
    if s is not None and len(s["ibi_ms"]):
        srcs["samsung"] = af_screen.segments_from_intervals(s["ibi_end_ms"], s["ibi_ms"], s["ibi_ok"], s["ibi_adjacent"])
    sess = [(x.start_ms, x.end_ms, bench.GROUPS.get(x.kind, x.kind)) for x in rec.sessions]
    rows = []
    for name, segs in srcs.items():
        n_alerts = len(af_screen.alerts(segs))
        for k, sg in enumerate(segs):
            mid = (sg.start_ms + sg.end_ms) / 2
            group = next((g for a, b, g in sess if a <= mid < b), "between sessions")
            rows.append({"uid": uid, "source": name, "k": k, "group": group, "minutes": (sg.end_ms - sg.start_ms) / 60000,
                         "nrmssd": sg.nrmssd, "shannon": sg.shannon, "irregular": sg.irregular,
                         "alerts_in_recording": n_alerts})
    return rows


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/benchmarks/af25"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    uids = [u for u in galaxyppg.UIDS if u != "P01"]
    galaxyppg.prepare_all(uids, workers=a.workers)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for rs in ex.map(participant, uids) for r in rs]
    df = pl.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(a.out / "segments.parquet")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(200)
    by = df.group_by("source").agg(
        people=pl.col("uid").n_unique(), segments=pl.len(), hours=(pl.col("minutes").sum() / 60).round(1),
        irregular_pct=(pl.col("irregular").mean() * 100).round(2),
        nrmssd_p95=pl.col("nrmssd").quantile(0.95).round(3), shannon_p95=pl.col("shannon").quantile(0.95).round(3),
    ).sort("source")
    alerts = df.group_by("source", "uid").agg(pl.col("alerts_in_recording").first()).group_by("source").agg(
        people_with_alerts=(pl.col("alerts_in_recording") > 0).sum(), alerts=pl.col("alerts_in_recording").sum()).sort("source")
    groups = df.group_by("source", "group").agg(segments=pl.len(), irregular_pct=(pl.col("irregular").mean() * 100).round(2)).sort("source", "group")
    print("False positives per 64-beat segment (everyone is in sinus rhythm)")
    print(by)
    print(alerts)
    print(groups)
    for name, t in (("summary", by), ("alerts", alerts), ("groups", groups)):
        t.write_csv(a.out / f"{name}.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
