"""Phase 0 check: 25 Hz PPG beat timing against the watch's own ECG, recorded together.

A run recorded with `--es cmd record --es sensors sdk_ecg,hr_raw,acc` (and optionally sdk_ppg) holds
the Samsung SDK ECG (500 Hz, finger on the top key) next to the 25 Hz PPG streams. R-peaks come from
`watch_session._ecg_peaks` and PPG beats from `watch_session._hrv25_beats` (foot_template, exactly as
the server computes 25 Hz HRV), so this scores the production path.

The scoring window is, like the benchmark's usable windows, one where the reference is clean and the
PPG is present: the longest run of consecutive clean ECG intervals (the first seconds after the finger
touches the key are unusable while the electrode settles), cut to the time the PPG was recording
(the ECG keeps running while the watch flushes the SDK batch). Inside it, with the GalaxyPPG
benchmark's own matching (`bench.match_beats`, +-50 ms after removing a constant pulse-arrival delay):
- precision / recall of the beats foot_template keeps, beat timing SD, IBI MAE and bias over
  consecutive matched beats;
- RMSSD/SDNN of the PPG's kept intervals against the ECG's clean intervals in the window, as the
  benchmark scores a window (and as the server computes HRV); RMSSD over the matched intervals only
  (same beats from both sources); and the server's whole-run values for reference.

Gate, fixed before the first recording was scored: IBI MAE and timing SD at or below the 90th
percentile of held-out GalaxyPPG seated-rest 60 s windows (foot_template), precision and recall at or
above the 10th, and the window RMSSD error inside the benchmark's 60 s Bland-Altman limits. (The first
version compared the server's whole-run PPG RMSSD with the ECG's clean stretch, two different
windows; the benchmark always compares one window.)

    uv run python benchmarks/watch_ecg/run.py [RUN_DIR ...] [--out data/benchmarks/watch_ecg]

Without run directories it scores every run in data/runs with sdk_ecg.csv and a 25 Hz PPG CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pulse_exploration import watch_session as ws  # noqa: E402
from pulse_exploration.hrv import bench, time_domain  # noqa: E402

RUNS = Path("data/runs")
BENCH_WINDOWS = Path("data/benchmarks/hrv25/windows.parquet")
TOL_MS = bench.BenchConfig().match_tol_ms


def benchmark_limits() -> dict:
    """Held-out foot_template seated-rest 60 s windows: percentiles and RMSSD Bland-Altman limits."""
    w = pl.read_parquet(BENCH_WINDOWS).filter(
        (pl.col("estimator") == "foot_template") & (pl.col("window_s") == 60)
        & (pl.col("group") == "seated-rest") & (pl.col("split") == "held_out"))
    out = {"windows": w.height}
    for col in ("precision", "recall", "ibi_mae_ms", "timing_sd_ms"):
        v = w[col].drop_nulls().drop_nans().to_numpy()
        out[col] = {"p10": float(np.percentile(v, 10)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                    "values": v}
    c = w.filter(pl.col("covered"))
    err = (c["est_rmssd"] - c["ref_rmssd"]).to_numpy()
    out["rmssd_error"] = {"bias": float(err.mean()), "loa_low": float(err.mean() - 1.96 * err.std(ddof=1)),
                          "loa_high": float(err.mean() + 1.96 * err.std(ddof=1)),
                          "median_abs": float(np.median(np.abs(err))), "windows": c.height}
    return out


def score_source(alias: str, tables: dict, clock, ecg) -> dict:
    t, x, valid, _ = ws._ppg_input(alias, tables[alias], clock)
    beats, why = ws._hrv25_beats(t, x, valid, ws._acc_25hz(ws._accelerometer(tables, clock)))
    if beats is None:
        return {"error": why}
    beat_ms = beats.t * 1000
    est = beat_ms[beats.good]
    # the longest run of clean ECG intervals: R-peaks s .. e
    s, e = max(ws._runs(ecg.ok), key=lambda r: r[1] - r[0])
    ref, ok = ecg.t[s:e + 1] * 1000, ecg.ok[s:e]
    delay = bench.match_beats(est, ref, TOL_MS, delay_from=(est >= ref[0]) & (est <= ref[-1]))["delay_ms"]
    # cut to R-peaks whose pulse falls inside the PPG recording
    inside = np.flatnonzero((ref + delay >= beats.t_first * 1000 + TOL_MS) & (ref + delay <= beats.t_last * 1000 - TOL_MS))
    ref, ok = ref[inside[0]:inside[-1] + 1], ok[inside[0]:inside[-1]]
    w0, w1 = ref[0] - TOL_MS, ref[-1] + TOL_MS
    est = est[(est - delay >= w0) & (est - delay <= w1)]
    mt = bench.match_beats(est, ref, TOL_MS)
    pairs = mt["pairs"]
    pe, pr = pairs[:, 0], pairs[:, 1]
    # HRV in the window, as the benchmark scores it: PPG kept intervals vs ECG clean intervals
    kept = beats.good & (beat_ms - delay >= w0) & (beat_ms - delay <= w1)
    ppg_hrv = ws._hrv_values(ws._segments(np.diff(beat_ms), kept[:-1] & kept[1:]))
    ecg_hrv = ws._hrv_values(ws._segments(np.diff(ref), ok))
    out = {"beats_kept": int(beats.good.sum()), "beats_rejected": int((~beats.good).sum()),
           "lost_samples_corrected": beats.lost_samples, "fs_hz": beats.fs_hz, "delay_ms": mt["delay_ms"],
           "window_s": [float(ref[0] / 1000), float(ref[-1] / 1000)],
           "precision": float(len(pe) / len(est)), "recall": float(len(pr) / len(ref)),
           "matched": int(len(pairs)), "ecg_beats": int(len(ref)), "timing_sd_ms": float(np.std(mt["resid"])),
           "window_hrv": {"ppg": ppg_hrv, "ecg": ecg_hrv},
           "rmssd_error_ms": ppg_hrv["rmssd_ms"] - ecg_hrv["rmssd_ms"] if ppg_hrv and ecg_hrv else None}
    cons = (np.diff(pe) == 1) & (np.diff(pr) == 1) & ok[np.minimum(pr[:-1], len(ok) - 1)]
    est_ibi = (est[pe[1:]] - est[pe[:-1]])[cons]
    ref_ibi = (ref[pr[1:]] - ref[pr[:-1]])[cons]
    err = est_ibi - ref_ibi
    adjacent = (np.flatnonzero(cons)[1:] - np.flatnonzero(cons)[:-1]) == 1
    paired_est, paired_ref = time_domain(est_ibi, adjacent=adjacent), time_domain(ref_ibi, adjacent=adjacent)
    out.update({"ibi_pairs": int(cons.sum()), "ibi_mae_ms": float(np.abs(err).mean()), "ibi_bias_ms": float(err.mean()),
                "ibi_max_abs_ms": float(np.abs(err).max()),
                "paired_rmssd_ms": {"ppg": paired_est["rmssd"], "ecg": paired_ref["rmssd"],
                                    "successive_diffs": paired_est["n_diff"]},
                "ibi_errors_ms": [round(float(v), 1) for v in err]})
    return out


def pct_rank(values: np.ndarray, x: float) -> float:
    return float(100 * np.mean(values <= x))


def score_run(run: Path, limits: dict) -> dict:
    tables = ws.load_tables(run)
    clock = ws._Clock(ws._read_kv(run / "meta.txt"), tables)
    ecg, why = ws._ecg_peaks(tables["sdk_ecg"], clock)
    if ecg is None:
        return {"run": run.name, "error": f"ECG: {why}"}
    analysis = ws.analyze_run(run, write=False)
    hrv = {e["source"]: e for e in analysis["metrics"]["hrv"]["estimates"]}
    ecg_hrv = hrv.get("sdk_ecg")
    res = {"run": run.name, "ecg": {"lead_on_s": ecg.lead_on_s, "r_peaks": int(len(ecg.t)),
                                    "clean_intervals": int(ecg.ok.sum()), "inverted": ecg.inverted,
                                    "span_s": [float(ecg.t[0]), float(ecg.t[-1])],
                                    "hrv": ecg_hrv and ecg_hrv["value"],
                                    "reliability": ecg_hrv and ecg_hrv["reliability"]},
           "sources": {}}
    for alias in ws.HRV25_SOURCES:
        if alias not in tables:
            continue
        s = score_source(alias, tables, clock, ecg)
        if "error" not in s:
            server = hrv.get(alias)
            s["server_hrv"] = server and server["value"]
            s["server_reliability"] = server and server["reliability"]
            checks = {
                "ibi_mae_ms": s["ibi_mae_ms"] <= limits["ibi_mae_ms"]["p90"],
                "timing_sd_ms": s["timing_sd_ms"] <= limits["timing_sd_ms"]["p90"],
                "precision": s["precision"] >= limits["precision"]["p10"],
                "recall": s["recall"] >= limits["recall"]["p10"],
                "rmssd_error_ms": (s["rmssd_error_ms"] is not None
                                   and limits["rmssd_error"]["loa_low"] <= s["rmssd_error_ms"] <= limits["rmssd_error"]["loa_high"]),
            }
            s["percentile_in_benchmark"] = {k: pct_rank(limits[k]["values"], s[k])
                                            for k in ("ibi_mae_ms", "timing_sd_ms", "precision", "recall")}
            s["checks"] = checks
            s["passed"] = all(checks.values())
        res["sources"][alias] = s
    return res


def report(results: list[dict], limits: dict) -> str:
    rows = ["| Run | Source | Window (s) | Beats matched | Precision | Recall | Timing SD (ms) | IBI MAE (ms) "
            "| IBI bias (ms) | RMSSD PPG vs ECG (ms) | Paired RMSSD (ms) | Server, whole run (ms) | Passed |", "|---" * 13 + "|"]
    nan = float("nan")
    for r in results:
        if "error" in r:
            rows.append(f"| {r['run']} | — | {r['error']} |" + " |" * 10)
            continue
        for alias, s in r["sources"].items():
            if "error" in s:
                rows.append(f"| {r['run']} | {alias} | {s['error']} |" + " |" * 10)
                continue
            wh, pr = s["window_hrv"], s["paired_rmssd_ms"]
            server = s["server_hrv"]["rmssd_ms"] if s["server_hrv"] else nan
            ecg = r["ecg"]["hrv"]["rmssd_ms"] if r["ecg"]["hrv"] else nan
            rows.append(f"| {r['run']} | {alias} | {s['window_s'][0]:.1f}–{s['window_s'][1]:.1f} "
                        f"| {s['matched']}/{s['ecg_beats']} | {s['precision']:.2f} | {s['recall']:.2f} "
                        f"| {s['timing_sd_ms']:.1f} | {s['ibi_mae_ms']:.1f} | {s['ibi_bias_ms']:+.1f} "
                        f"| {wh['ppg']['rmssd_ms']:.1f} vs {wh['ecg']['rmssd_ms']:.1f} | {pr['ppg']:.1f} vs {pr['ecg']:.1f} "
                        f"| {server:.1f} vs {ecg:.1f} "
                        f"| {'yes' if s['passed'] else 'no: ' + ', '.join(k for k, v in s['checks'].items() if not v)} |")
    lim = (f"Benchmark (held-out GalaxyPPG foot_template, seated rest, {limits['windows']} windows of 60 s): "
           f"precision p10 {limits['precision']['p10']:.2f}, recall p10 {limits['recall']['p10']:.2f}, "
           f"timing SD median {limits['timing_sd_ms']['median']:.1f} / p90 {limits['timing_sd_ms']['p90']:.1f} ms, "
           f"IBI MAE median {limits['ibi_mae_ms']['median']:.1f} / p90 {limits['ibi_mae_ms']['p90']:.1f} ms, "
           f"RMSSD error {limits['rmssd_error']['bias']:+.1f} ms ({limits['rmssd_error']['loa_low']:+.1f} to "
           f"{limits['rmssd_error']['loa_high']:+.1f}).")
    return "\n".join(rows) + "\n\n" + lim + "\n"


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/benchmarks/watch_ecg"))
    args = ap.parse_args(argv)
    runs = args.runs or sorted(p for p in RUNS.iterdir()
                               if (p / "sdk_ecg.csv").is_file() and any((p / f"{a}.csv").is_file() for a in ws.HRV25_SOURCES))
    limits = benchmark_limits()
    results = [score_run(r, limits) for r in runs]
    args.out.mkdir(parents=True, exist_ok=True)
    plain = {k: {kk: vv for kk, vv in v.items() if kk != "values"} if isinstance(v, dict) else v for k, v in limits.items()}
    (args.out / "results.json").write_text(json.dumps({"limits": plain, "runs": results}, indent=1, default=float) + "\n")
    text = report(results, limits)
    (args.out / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
