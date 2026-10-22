"""Watch run analysis (watch_session, watch_api) on synthetic run directories, plus a smoke run over
the real recordings in data/runs/ when they exist (values printed, not asserted; the real runs are
copied to a temporary directory so the tests never write into data/).

Run: ``uv run pytest -q tests/test_watch_session.py`` (add ``-s`` to see the real-run printout).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watch_synth as ws

from pulse_exploration import watch_api, watch_raw
from pulse_exploration import watch_session as w

ROOT = Path(__file__).resolve().parents[1]
REAL_RUNS = ("20261022-013105-record", "20261022-015638-record", "20261022-015808-record")
ESTIMATE_KEYS = {"value", "source", "feature", "tier", "reliability", "n", "notes", "details"}
HRV_KEYS = {"rmssd_ms", "sdnn_ms", "pnn50", "mean_nn_ms", "n_beats", "window_s"}
STILL = (1.0, 90.0)  # synthetic runs: on the wrist from 1 s, motion burst from 90 s

SPECS = {
    "basic": ("20261022-100001-record", dict(tier="25hz", motion=[(90, 120)], seed=1)),
    "full": ("20261022-100002-record", dict(tier="all", motion=[(90, 120)], ecg_s=(10, 40), seed=2)),
    "legacy": ("20261022-100003-record", dict(tier="all", motion=[(90, 120)], with_capabilities=False, seed=4)),
    # 3 s of motion every 8 s: no motion-free stretch long enough for HRV
    "moving": ("20261022-100004-record", dict(tier="25hz", motion=[(k * 8 + 5, k * 8 + 8) for k in range(15)],
                                              seed=3)),
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("watch_runs")
    out = {}
    for key, (name, kw) in SPECS.items():
        run, truth = ws.make_run(root, name, **kw)
        out[key] = (run, truth, w.analyze_run(run))
    return out


def est(res: dict, metric: str, source: str) -> dict:
    found = [e for e in res["metrics"][metric]["estimates"] if e["source"] == source]
    assert found, f"no {metric} estimate from {source}: {res['metrics'][metric]}"
    return found[0]


def codes(res: dict) -> set[str]:
    return {x["code"] for x in res["warnings"]}


def copy_run(run: Path, dst_root: Path, drop: tuple[str, ...] = ()) -> Path:
    dst = dst_root / run.name
    shutil.copytree(run, dst)
    for name in (*drop, w.ANALYSIS_FILE):
        (dst / name).unlink(missing_ok=True)
    return dst


def truth_rmssd_in_spans(truth: ws.WatchTruth, spans: list[list[float]], pulse: bool) -> float:
    t = truth.pulse_s if pulse else truth.r_peaks_s
    diffs = []
    for a, b in spans:
        nn = np.diff(t[(t >= a - 0.05) & (t <= b + 0.05)]) * 1000
        diffs.extend(np.diff(nn))
    return float(np.sqrt(np.mean(np.square(diffs))))


# --------------------------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------------------------

def test_json_shape_and_file(runs):
    for run, _, res in runs.values():
        on_disk = json.loads((run / w.ANALYSIS_FILE).read_text())
        assert on_disk == res
        json.dumps(res, allow_nan=False)
        for key in ("tier", "capabilities", "capabilities_source", "features_used", "summary", "metrics",
                    "warnings", "sources", "duration_s"):
            assert key in res
        assert set(res["metrics"]) == {"hr", "hrv", "motion", "on_wrist", "skin_temp", "spo2"}
        for metric in res["metrics"].values():
            assert set(metric) == {"chosen", "estimates", "skipped"}
            for e in metric["estimates"]:
                assert set(e) == ESTIMATE_KEYS
                assert e["reliability"] in w.RELIABILITY
                assert e["tier"] == ("all" if e["source"].startswith("sdk_") else "25hz")
            if metric["estimates"]:
                assert metric["chosen"] in metric["estimates"]
        for e in res["metrics"]["hrv"]["estimates"]:
            assert set(e["value"]) == HRV_KEYS


# --------------------------------------------------------------------------------------------
# tier 25hz
# --------------------------------------------------------------------------------------------

def test_basic_tier_sources(runs):
    _, truth, res = runs["basic"]
    assert res["tier"] == "25hz" and res["capabilities_source"] == "file"
    assert [e["source"] for e in res["metrics"]["hr"]["estimates"]] == ["hr", "hr_raw"]
    hr = res["metrics"]["hr"]["chosen"]
    assert hr["source"] == "hr" and hr["reliability"] == "good"
    assert abs(hr["value"] - truth.hr_bpm(*STILL)) <= 2
    ppg_hr = est(res, "hr", "hr_raw")
    assert abs(ppg_hr["details"]["peaks_bpm"] - truth.hr_bpm(*STILL)) <= 2
    assert abs(ppg_hr["details"]["spectrum_bpm"] - truth.hr_bpm(*STILL)) <= 3
    assert [e["source"] for e in res["metrics"]["hrv"]["estimates"]] == ["hr_raw"]
    assert res["features_used"] == ["ppg", "hr", "motion"]
    assert not any(e["tier"] == "all" for m in res["metrics"].values() for e in m["estimates"])
    assert any("developer mode" in x["message"] for x in res["warnings"] if x["code"] == "source_error")
    assert 0.98 <= res["summary"]["on_wrist_share"] < 1.0  # first second off the wrist


def test_25hz_hrv_experimental_from_motion_free_windows(runs):
    _, truth, res = runs["basic"]
    hrv = res["metrics"]["hrv"]["chosen"]
    assert hrv["source"] == "hr_raw" and hrv["feature"] == "ppg"
    assert hrv["reliability"] == "experimental"
    gate = hrv["details"]["gate"]
    assert gate["passed"] and gate["motion_checked"] and gate["method"] == "foot_template"
    assert gate["beats_rejected"] > 0
    # two ~60 s windows: the second is half motion (90-120 s) and keeps too few intervals
    assert [x["accepted"] for x in gate["windows"]] == [True, False]
    spans = hrv["details"]["spans_s"]
    for a, b in spans:  # nothing from the motion burst or its padding
        assert not truth.in_motion(np.linspace(a, b, 50), pad=w.MOTION_PAD_S).any(), (a, b)
        assert b <= 90 - w.MOTION_PAD_S
    assert 45 <= hrv["value"]["window_s"] <= 90
    truth_rmssd = truth_rmssd_in_spans(truth, spans, pulse=True)
    print(f"\n25 Hz RMSSD {hrv['value']['rmssd_ms']} ms vs truth {truth_rmssd:.1f} ms")
    assert abs(hrv["value"]["rmssd_ms"] - truth_rmssd) <= 10
    assert "motion" in codes(res)


def test_motion_leaves_no_usable_25hz_hrv(runs):
    _, truth, res = runs["moving"]
    found = [e for e in res["metrics"]["hrv"]["estimates"] if e["source"] == "hr_raw"]
    assert all(e["reliability"] == "unreliable" for e in found)
    chosen = res["metrics"]["hrv"]["chosen"]
    assert chosen is None or chosen["reliability"] == "unreliable"
    assert "motion" in codes(res) and codes(res) & {"no_hrv", "hrv_unreliable"}
    if not found:  # foot_template kept too few beats: the reason is reported
        assert any(s["source"] == "hr_raw" for s in res["metrics"]["hrv"]["skipped"])
    for e in found:
        for a, b in e["details"]["spans_s"]:  # still only motion-free intervals
            assert not truth.in_motion(np.linspace(a, b, 20)).any(), (a, b)


def test_no_accelerometer_means_unreliable(runs, tmp_path):
    run = copy_run(runs["basic"][0], tmp_path, drop=("acc.csv", "gyro.csv"))
    res = w.analyze_run(run)
    e = est(res, "hrv", "hr_raw")
    assert e["reliability"] == "unreliable"
    assert any("no accelerometer" in r for r in e["details"]["gate"]["reasons"])
    assert res["metrics"]["motion"]["chosen"] is None


# --------------------------------------------------------------------------------------------
# tier all
# --------------------------------------------------------------------------------------------

def test_full_tier_priority(runs):
    _, truth, res = runs["full"]
    assert res["tier"] == "all"
    hrv = res["metrics"]["hrv"]
    assert [e["source"] for e in hrv["estimates"]] == ["sdk_ecg", "sdk_ppg", "sdk_hr", "hr_raw"]
    assert hrv["chosen"]["source"] == "sdk_ecg" and hrv["chosen"]["reliability"] == "reference"
    assert est(res, "hrv", "sdk_hr")["reliability"] == "good"
    assert est(res, "hrv", "sdk_hr")["feature"] == "ibi"
    assert est(res, "hrv", "sdk_ppg")["reliability"] == "good"  # the benchmarked source
    assert est(res, "hrv", "hr_raw")["reliability"] == "experimental"  # not yet checked against the ECG
    for src in ("sdk_ppg", "hr_raw"):  # the per-axis gate keeps the 90-120 s wrist shake out
        for a, b in est(res, "hrv", src)["details"]["spans_s"]:
            assert not truth.in_motion(np.linspace(a, b, 50)).any(), (src, a, b)
    assert "hrv_sources_disagree" not in codes(res)
    hr = res["metrics"]["hr"]
    assert hr["chosen"]["source"] == "sdk_hr"
    assert {e["source"] for e in hr["estimates"]} == {"sdk_hr", "hr", "sdk_ecg", "sdk_ppg", "hr_raw"}
    assert est(res, "hr", "sdk_ecg")["reliability"] == "reference"
    for e in hr["estimates"]:
        assert abs(e["value"] - truth.hr_bpm(*STILL)) <= 3, e
    assert res["features_used"] == ["ppg", "hr", "ibi", "motion", "ecg"]


def test_ecg_hrv_matches_truth(runs):
    _, truth, res = runs["full"]
    e = est(res, "hrv", "sdk_ecg")
    a, b = truth.ecg_s
    true_rmssd = truth.rmssd(a, b)
    print(f"\nECG RMSSD {e['value']['rmssd_ms']} ms vs truth {true_rmssd:.2f} ms; "
          f"SDNN {e['value']['sdnn_ms']} vs {np.std(truth.intervals(a, b), ddof=1):.2f}")
    assert abs(e["value"]["rmssd_ms"] - true_rmssd) <= 5
    # over exactly the stretch it used (beats next to lead-off and the record end are excluded)
    assert abs(e["value"]["rmssd_ms"] - truth_rmssd_in_spans(truth, e["details"]["spans_s"], pulse=False)) <= 1
    assert abs(e["value"]["mean_nn_ms"] - truth.intervals(a, b).mean()) <= 3
    assert e["details"]["lead_on_s"] >= w.ECG_REFERENCE_S
    assert abs(e["value"]["n_beats"] - (len(truth.intervals(a, b)) + 1)) <= 2


def test_samsung_ibi_lost_sample_interval_dropped():
    """A smooth rhythm (60 bpm, slow 20 ms swing) where one listed interval is a sample period
    (40 ms) short, the error a silently lost PPG sample causes: that interval is dropped and RMSSD
    comes back to the smooth rhythm's."""
    import polars as pl

    n = 90
    ibi = 1000 + 20 * np.sin(2 * np.pi * np.arange(n) / 12)
    clean_rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    lost = ibi.copy()
    lost[45] -= 40.0
    t0 = 1_792_700_000_000
    ends = t0 + np.cumsum(ibi)
    rows = {"timestamp_ms": [], "wall_ms": [], "hr": [], "hr_status": [], "ibi_ms": [], "ibi_status": []}
    for k in range(n):  # one interval per 1 s row
        rows["timestamp_ms"].append(int(ends[k]))
        rows["wall_ms"].append(int(ends[k]) + 500)
        rows["hr"].append(60)
        rows["hr_status"].append(1)
        rows["ibi_ms"].append(str(int(round(lost[k]))))
        rows["ibi_status"].append("0")
    df = pl.DataFrame(rows)
    clock = w._Clock({"start_wall_ms": str(t0)}, {"sdk_hr": df})
    e, why = w._samsung_ibi(df, clock, w._Motion(None))
    assert why is None
    assert e["details"]["dropped_lost_sample"] == 1
    assert abs(e["value"]["rmssd_ms"] - clean_rmssd) < 1.0
    assert e["reliability"] == "experimental"  # no accelerometer: motion not checked


def test_samsung_ibi_filtering(runs):
    _, truth, res = runs["full"]
    e = est(res, "hrv", "sdk_hr")
    d = e["details"]
    assert d["dropped_status"] >= 1  # the injected status -1 IBI (and motion rows)
    assert d["row_gaps"] >= 1  # the missing row breaks adjacency
    assert d["segments"] >= 2
    assert d["dropped_motion"] > 0
    assert e["value"]["window_s"] < 90
    true_rmssd = truth.rmssd(*STILL, pulse=True)
    assert abs(e["value"]["rmssd_ms"] - true_rmssd) <= 8
    assert any("no beat times" in n for n in e["notes"])


def test_sdk_ppg_then_ibi_chosen_without_ecg(runs, tmp_path):
    run = copy_run(runs["full"][0], tmp_path, drop=("sdk_ecg.csv",))
    caps = json.loads((run / w.CAPABILITIES_FILE).read_text())
    caps["features"]["ecg"] = {"available": False, "tier": None, "sources": [], "reason": "not recorded in this run"}
    (run / w.CAPABILITIES_FILE).write_text(json.dumps(caps))
    res = w.analyze_run(run)
    assert res["metrics"]["hrv"]["chosen"]["source"] == "sdk_ppg"
    assert res["metrics"]["hrv"]["chosen"]["reliability"] == "good"
    assert res["metrics"]["hr"]["chosen"]["source"] == "sdk_hr"
    (run / "sdk_ppg.csv").unlink()
    res = w.analyze_run(run)
    assert res["metrics"]["hrv"]["chosen"]["source"] == "sdk_hr"
    assert res["metrics"]["hrv"]["chosen"]["reliability"] == "good"


# --------------------------------------------------------------------------------------------
# runs without capabilities.json
# --------------------------------------------------------------------------------------------

def test_inferred_capabilities_all(runs):
    _, _, res = runs["legacy"]
    assert res["capabilities_source"] == "inferred" and "capabilities_inferred" in codes(res)
    caps = res["capabilities"]
    assert caps["schema"] == 1 and caps["tier"] == "all" and res["tier"] == "all"
    assert caps["sdk"]["policy"] == "allowed"
    f = caps["features"]
    assert f["ppg"]["sources"] == ["sdk_ppg", "hr_raw"] and f["ppg"]["tier"] == "all"
    assert f["hr"]["sources"] == ["sdk_hr", "hr"]
    assert f["ibi"]["available"] and f["ibi"]["sources"] == ["sdk_hr"]
    assert f["motion"]["sources"] == ["acc", "gyro"] and f["motion"]["tier"] == "25hz"
    assert not f["ecg"]["available"] and f["ecg"]["reason"] == "not recorded in this run"
    assert not f["ppg100"]["available"]
    assert res["metrics"]["hrv"]["chosen"]["source"] == "sdk_ppg"
    assert res["metrics"]["hr"]["chosen"]["source"] == "sdk_hr"


def test_inferred_capabilities_25hz(runs, tmp_path):
    run = copy_run(runs["basic"][0], tmp_path, drop=(w.CAPABILITIES_FILE,))
    res = w.analyze_run(run)
    caps = res["capabilities"]
    assert res["capabilities_source"] == "inferred" and caps["tier"] == "25hz"
    assert caps["sdk"]["policy"] == "denied"  # SDK_POLICY_ERROR rows in record.tsv
    assert caps["features"]["ppg"]["sources"] == ["hr_raw"]
    assert caps["features"]["ibi"]["reason"] == w.POLICY_REASON
    assert caps["android"] == {"hr_raw": True, "hr": True, "acc": True, "gyro": True}
    assert res["metrics"]["hrv"]["chosen"]["source"] == "hr_raw"


def test_empty_run_does_not_crash(tmp_path):
    run = tmp_path / "20261022-100009-probe"
    run.mkdir()
    (run / "meta.txt").write_text("seconds=8\n")
    (run / "sdk_hr.csv").write_text("")
    res = w.analyze_run(run)
    assert res["metrics"]["hr"]["chosen"] is None and res["metrics"]["hrv"]["chosen"] is None
    assert {"no_data", "too_short"} <= codes(res)


# --------------------------------------------------------------------------------------------
# HTTP API and the watch_raw CLI
# --------------------------------------------------------------------------------------------

def test_api_analysis_cache_and_refresh(runs, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pulse_exploration import server

    run = copy_run(runs["basic"][0], tmp_path)
    monkeypatch.setattr(watch_api, "DATA", tmp_path)
    client = TestClient(server.app)
    url = f"/runs/{run.name}/analysis"
    r = client.get(url)
    assert r.status_code == 200 and r.json()["tier"] == "25hz"
    cache = run / w.ANALYSIS_FILE
    assert cache.is_file()
    marked = json.loads(cache.read_text()) | {"marker": 1}
    cache.write_text(json.dumps(marked))
    assert client.get(url).json().get("marker") == 1  # served from analysis.json
    assert "marker" not in client.get(url, params={"refresh": 1}).json()
    cache.write_text(json.dumps(marked))
    later = cache.stat().st_mtime + 10
    os.utime(run / "hr.csv", (later, later))  # a newer upload invalidates the cache
    assert "marker" not in client.get(url).json()
    assert client.get("/runs/.hidden/analysis").status_code == 400
    assert client.get("/runs/bad%20name/analysis").status_code == 400
    assert client.get("/runs/20990101-000000-record/analysis").status_code == 404


def test_watch_raw_cli_still_works(tmp_path):
    # watch_raw is the older whole-recording pipeline (no motion gate), so give it a still run
    run, truth = ws.make_run(tmp_path, "20261022-100005-record", tier="25hz", duration_s=60, seed=5)
    res = watch_raw.analyse(run)  # empty sdk_*.csv files present
    assert abs(res["hr_fft_bpm"] - truth.hr_bpm(1, 60)) <= 3
    proc = subprocess.run([sys.executable, "-m", "pulse_exploration.watch_raw", str(run)], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert (run / "ppg.png").is_file()


# --------------------------------------------------------------------------------------------
# real recordings (values printed only)
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", REAL_RUNS)
def test_real_runs_do_not_crash(name, tmp_path):
    src = ROOT / "data" / "runs" / name
    if not src.is_dir():
        pytest.skip(f"{name} not present")
    run = copy_run(src, tmp_path)
    res = w.analyze_run(run)
    json.dumps(res, allow_nan=False)
    hr, hrv = res["metrics"]["hr"]["chosen"], res["metrics"]["hrv"]["chosen"]
    print(f"\n== {name}: tier {res['tier']} ({res['capabilities_source']}), features {res['features_used']}")
    for e in res["metrics"]["hr"]["estimates"]:
        print(f"   hr  {e['source']:<10} {e['value']:>6} bpm  {e['reliability']:<12} n={e['n']}"
              + ("  <- chosen" if e is hr or e == hr else ""))
    for e in res["metrics"]["hrv"]["estimates"]:
        v = e["value"]
        print(f"   hrv {e['source']:<10} RMSSD {v['rmssd_ms']} SDNN {v['sdnn_ms']} pNN50 {v['pnn50']} "
              f"meanNN {v['mean_nn_ms']} beats {v['n_beats']} window {v['window_s']} s  {e['reliability']}"
              + ("  <- chosen" if e == hrv else ""))
    for s in res["metrics"]["hrv"]["skipped"]:
        print(f"   hrv skipped {s['source']}: {s['reason']}")
    print("   warnings:", "; ".join(f"{x['code']}: {x['message']}" for x in res["warnings"]))
