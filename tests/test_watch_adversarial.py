"""Adversarial cases for watch_session.analyze_run, plus a cross-check against the exact file layout
the Kotlin probe writes (Sources.kt, SamsungSdk.kt, Capabilities.kt).

Run: ``uv run pytest -q tests/test_watch_adversarial.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watch_synth as ws

from pulse_exploration import watch_raw
from pulse_exploration import watch_session as w

STILL = (1.0, 90.0)


def est(res: dict, metric: str, source: str) -> dict:
    found = [e for e in res["metrics"][metric]["estimates"] if e["source"] == source]
    assert found, f"no {metric} estimate from {source}: {res['metrics'][metric]}"
    return found[0]


def codes(res: dict) -> set[str]:
    return {x["code"] for x in res["warnings"]}


def truth_rmssd_in_spans(t: np.ndarray, spans: list[list[float]]) -> float:
    diffs = []
    for a, b in spans:
        nn = np.diff(t[(t >= a - 0.05) & (t <= b + 0.05)]) * 1000
        diffs.extend(np.diff(nn))
    return float(np.sqrt(np.mean(np.square(diffs))))


def crosses(spans: list[list[float]], t: float, margin: float = 0.1) -> bool:
    return any(a < t - margin and b > t + margin for a, b in spans)


def read(run: Path, alias: str) -> pl.DataFrame:
    df = watch_raw.read_csv(run / f"{alias}.csv")
    assert df is not None
    return df


def write(run: Path, alias: str, df: pl.DataFrame) -> None:
    df.write_csv(run / f"{alias}.csv")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("adv_base")
    out = {}
    for key, kw in {"basic": dict(tier="25hz", motion=[(90, 120)], seed=1),
                    "full": dict(tier="all", motion=[(90, 120)], seed=2),
                    "ecg": dict(tier="all", motion=[(90, 120)], ecg_s=(10, 40), seed=2)}.items():
        run, truth = ws.make_run(root, f"20261022-1100{len(out):02d}-record", **kw)
        out[key] = (run, truth, w.analyze_run(run, write=False))
    return out


def clone(base_run: Path, tmp_path: Path) -> Path:
    dst = tmp_path / base_run.name
    shutil.copytree(base_run, dst)
    return dst


# --------------------------------------------------------------------------------------------
# missed and extra beats
# --------------------------------------------------------------------------------------------

def _hr_raw_times(df: pl.DataFrame) -> np.ndarray:
    return df["event_ns"].to_numpy() / 1e9 - ws.BOOT_S


def test_ppg_missed_and_extra_beats(base, tmp_path):
    src, truth, _ = base["basic"]
    run = clone(src, tmp_path)
    df = read(run, "hr_raw")
    t = _hr_raw_times(df)
    p = truth.pulse_s
    missed = float(p[np.argmin(np.abs(p - 30))])
    k = int(np.argmin(np.abs(p - 50)))
    extra = float((p[k] + p[k + 1]) / 2)
    cols = {c: df[c].to_numpy().astype(float) for c in ("v1", "v4", "v5", "v7")}
    # missed beat: the pulse is flattened (straight line across it)
    m = (t >= missed - 0.35) & (t <= missed + 0.55)
    i0, i1 = np.flatnonzero(m)[[0, -1]]
    for c, x in cols.items():
        x[m] = np.interp(t[m], [t[i0 - 1], t[i1 + 1]], [x[i0 - 1], x[i1 + 1]])
    # extra beat: a full-size pulse half-way between two real ones
    bump = ws.ppg_wave(t, np.array([extra]))
    for c, gain in (("v1", 4000), ("v4", 4500), ("v5", 40000), ("v7", 40000)):
        cols[c] = cols[c] - gain * bump
    write(run, "hr_raw", df.with_columns([pl.Series(c, np.round(x).astype(np.int64)) for c, x in cols.items()]))
    res = w.analyze_run(run, write=False)
    e = est(res, "hrv", "hr_raw")
    assert e["reliability"] in ("experimental", "unreliable")  # 25 Hz: never above experimental
    spans = e["details"]["spans_s"]
    assert not crosses(spans, missed), (missed, spans)
    rm, tr = e["value"]["rmssd_ms"], truth_rmssd_in_spans(p, spans)
    print(f"\nmissed/extra PPG beats: RMSSD {rm} vs truth {tr:.1f}")
    assert abs(rm - tr) <= 10  # an accepted extra beat would add ~450 ms differences
    assert abs(est(res, "hr", "hr_raw")["value"] - truth.hr_bpm(*STILL)) <= 2


def _ibi_rows(df: pl.DataFrame) -> tuple[list[list[str]], list[list[str]]]:
    split = lambda v: [s for s in (v or "").split("|") if s]  # noqa: E731
    return [split(v) for v in df["ibi_ms"].to_list()], [split(v) for v in df["ibi_status"].to_list()]


def _with_ibi(df: pl.DataFrame, ibi: list[list[str]], st: list[list[str]]) -> pl.DataFrame:
    return df.with_columns(pl.Series("ibi_ms", ["|".join(v) for v in ibi], pl.String),
                           pl.Series("ibi_status", ["|".join(v) for v in st], pl.String))


def test_samsung_ibi_missed_beats_break_adjacency(base, tmp_path):
    src, truth, ref = base["full"]
    assert est(ref, "hrv", "sdk_hr")["details"]["suspected_dropped_beats"] == 0  # no false alarms
    run = clone(src, tmp_path)
    df = read(run, "sdk_hr")
    ibi, st = _ibi_rows(df)
    T = (df["timestamp_ms"].to_numpy() - ws.T0_MS) / 1000
    dropped = 0
    for r in range(len(ibi)):  # every ~8 s before the motion burst, a beat goes missing: rows stay 1 Hz
        if 5 < T[r] < 85 and r % 8 == 0 and ibi[r]:
            ibi[r], st[r] = ibi[r][1:], st[r][1:]
            dropped += 1
    write(run, "sdk_hr", _with_ibi(df, ibi, st))
    res = w.analyze_run(run, write=False)
    e = est(res, "hrv", "sdk_hr")
    found = e["details"]["suspected_dropped_beats"]
    print(f"\nIBI: {dropped} beats dropped, {found} detected; RMSSD {e['value']['rmssd_ms']} "
          f"vs {est(ref, 'hrv', 'sdk_hr')['value']['rmssd_ms']} before")
    # The drift heuristic only sees a drop once the elapsed time clearly outruns the listed intervals,
    # so drops followed soon by a two-beat row go unnoticed: it catches at least half, never invents one.
    assert dropped / 2 <= found <= dropped
    assert e["details"]["segments"] > est(ref, "hrv", "sdk_hr")["details"]["segments"]
    assert abs(e["value"]["rmssd_ms"] - truth.rmssd(*STILL, pulse=True)) <= 8


def test_samsung_ibi_extra_beat_dropped(base, tmp_path):
    src, _, ref = base["full"]
    run = clone(src, tmp_path)
    df = read(run, "sdk_hr")
    ibi, st = _ibi_rows(df)
    split_rows = [r for r in range(10, 80, 10) if len(ibi[r]) == 1 and st[r] == ["0"]]
    for r in split_rows:  # one interval reported as two halves (an extra detection)
        half = int(ibi[r][0]) // 2
        ibi[r], st[r] = [str(half), str(int(ibi[r][0]) - half)], ["0", "0"]
    write(run, "sdk_hr", _with_ibi(df, ibi, st))
    res = w.analyze_run(run, write=False)
    e, r0 = est(res, "hrv", "sdk_hr"), est(ref, "hrv", "sdk_hr")
    assert e["details"]["dropped_implausible"] >= r0["details"]["dropped_implausible"] + 2 * len(split_rows)
    assert e["details"]["suspected_dropped_beats"] == 0
    assert abs(e["value"]["rmssd_ms"] - r0["value"]["rmssd_ms"]) <= 8


def test_all_zero_ibi_lists(base, tmp_path):
    src, truth, _ = base["full"]
    run = clone(src, tmp_path)
    df = read(run, "sdk_hr")
    ibi, st = _ibi_rows(df)
    write(run, "sdk_hr", _with_ibi(df, [["0"] * len(v) for v in ibi], [["0"] * len(v) for v in ibi]))
    res = w.analyze_run(run, write=False)
    assert not [e for e in res["metrics"]["hrv"]["estimates"] if e["source"] == "sdk_hr"]
    skipped = {s["source"]: s["reason"] for s in res["metrics"]["hrv"]["skipped"]}
    assert "too few valid Samsung IBIs" in skipped["sdk_hr"]
    assert res["metrics"]["hrv"]["chosen"]["source"] == "sdk_ppg"  # the PPG does not depend on the IBI lists
    assert res["metrics"]["hr"]["chosen"]["source"] == "sdk_hr"  # HR rows are still fine
    assert abs(res["metrics"]["hr"]["chosen"]["value"] - truth.hr_bpm(*STILL)) <= 3


# --------------------------------------------------------------------------------------------
# ECG lead-off, hr_raw off-wrist
# --------------------------------------------------------------------------------------------

def _set_lead_off(run: Path, off: list[tuple[float, float]]) -> None:
    df = read(run, "sdk_ecg")
    T = (df["timestamp_ms"].to_numpy() - ws.T0_MS) / 1000
    m = np.zeros(len(T), bool)
    for a, b in off:
        m |= (T >= a) & (T < b)
    lead = np.where(m, 5, df["lead_off"].to_numpy())
    rng = np.random.default_rng(7)
    ecg = np.where(m, rng.normal(0, 0.8, len(T)), df["ecg_mv"].to_numpy())  # finger lifted: noise
    write(run, "sdk_ecg", df.with_columns(pl.Series("lead_off", lead), pl.Series("ecg_mv", np.round(ecg, 5))))


def test_ecg_lead_off_period(base, tmp_path):
    src, truth, ref = base["ecg"]
    run = clone(src, tmp_path)
    _set_lead_off(run, [(20.0, 23.0)])
    res = w.analyze_run(run, write=False)
    e = est(res, "hrv", "sdk_ecg")
    d = e["details"]
    assert abs(d["lead_on_s"] - (est(ref, "hrv", "sdk_ecg")["details"]["lead_on_s"] - 3)) <= 0.2
    assert e["reliability"] == "reference"  # still >= 20 s lead-on
    assert not crosses(d["spans_s"], 21.5, margin=1.4), d["spans_s"]
    rm, tr = e["value"]["rmssd_ms"], truth_rmssd_in_spans(truth.r_peaks_s, d["spans_s"])
    print(f"\nECG with lead-off 20-23 s: RMSSD {rm} vs truth {tr:.2f} over its spans")
    assert abs(rm - tr) <= 1.5
    assert res["metrics"]["hrv"]["chosen"]["source"] == "sdk_ecg"


def test_ecg_mostly_lead_off_falls_back(base, tmp_path):
    src, _, _ = base["ecg"]
    run = clone(src, tmp_path)
    _set_lead_off(run, [(10.0, 32.0)])  # only 8 s with the finger on the key
    res = w.analyze_run(run, write=False)
    ecg = [e for e in res["metrics"]["hrv"]["estimates"] if e["source"] == "sdk_ecg"]
    assert not ecg or ecg[0]["reliability"] == "unreliable"
    assert res["metrics"]["hrv"]["chosen"]["source"] == "sdk_ppg"  # next in line after the ECG


def _set_off_wrist(run: Path, a: float, b: float) -> None:
    df = read(run, "hr_raw")
    t = _hr_raw_times(df)
    m = (t >= a) & (t < b)
    rng = np.random.default_rng(3)
    repl = {"v0": np.where(m, 1, df["v0"].to_numpy())}
    for c in ("v1", "v4", "v5", "v7"):  # off the wrist: ambient light, nothing cardiac
        x = df[c].to_numpy().astype(float)
        repl[c] = np.round(np.where(m, ws.OFFSET + rng.normal(0, 30000, len(t)), x)).astype(np.int64)
    write(run, "hr_raw", df.with_columns([pl.Series(c, v) for c, v in repl.items()]))


def test_hr_raw_off_wrist_samples(base, tmp_path):
    src, truth, _ = base["basic"]
    run = clone(src, tmp_path)
    _set_off_wrist(run, 40.0, 55.0)
    res = w.analyze_run(run, write=False)
    assert res["summary"]["on_wrist_share"] < 0.9 and "off_wrist" in codes(res)
    e = est(res, "hrv", "hr_raw")
    for a, b in e["details"]["spans_s"]:
        assert b <= 40.05 or a >= 54.95, (a, b)
    assert e["reliability"] in ("experimental", "unreliable")
    assert abs(est(res, "hr", "hr_raw")["value"] - truth.hr_bpm(*STILL)) <= 2


def test_hr_raw_mostly_off_wrist_is_unreliable(base, tmp_path):
    src, _, _ = base["basic"]
    run = clone(src, tmp_path)
    _set_off_wrist(run, 16.0, 125.0)  # 15 s on the wrist
    res = w.analyze_run(run, write=False)
    e = [x for x in res["metrics"]["hrv"]["estimates"] if x["source"] == "hr_raw"]
    assert not e or e[0]["reliability"] == "unreliable"
    assert res["summary"]["on_wrist_share"] < 0.2


# --------------------------------------------------------------------------------------------
# sdk wall_ms is batch arrival
# --------------------------------------------------------------------------------------------

def test_sdk_ppg_uses_timestamp_not_batch_arrival(base, tmp_path):
    src, _, _ = base["full"]
    variants = {}
    for name, shift in (("same", None), ("late", 37_000)):
        run = clone(src, tmp_path / name)
        df = read(run, "sdk_ppg")
        wall = df["timestamp_ms"] if shift is None else (df["wall_ms"] + shift)
        write(run, "sdk_ppg", df.with_columns(wall.alias("wall_ms")))
        variants[name] = w.analyze_run(run, write=False)
    ref = w.analyze_run(src, write=False)
    for res in variants.values():
        for metric in ("hr", "hrv"):
            assert est(res, metric, "sdk_ppg")["value"] == est(ref, metric, "sdk_ppg")["value"]
        assert res["sources"]["sdk_ppg"] == ref["sources"]["sdk_ppg"]


# --------------------------------------------------------------------------------------------
# minimal runs
# --------------------------------------------------------------------------------------------

def test_only_hr_csv(base, tmp_path):
    src, truth, _ = base["basic"]
    run = tmp_path / "20261022-110010-record"
    run.mkdir()
    shutil.copy(src / "hr.csv", run / "hr.csv")
    shutil.copy(src / "meta.txt", run / "meta.txt")
    res = w.analyze_run(run)
    assert res["tier"] == "25hz" and res["capabilities_source"] == "inferred"
    hr = res["metrics"]["hr"]["chosen"]
    assert hr["source"] == "hr" and abs(hr["value"] - truth.hr_bpm(*STILL)) <= 2
    assert res["metrics"]["hrv"]["chosen"] is None and "no_hrv" in codes(res)
    assert res["metrics"]["motion"]["chosen"] is None and res["summary"]["on_wrist_share"] is None
    f = res["capabilities"]["features"]
    assert f["hr"]["sources"] == ["hr"] and not f["ppg"]["available"] and not f["motion"]["available"]
    assert res["features_used"] == ["hr"]
    json.dumps(res, allow_nan=False)


# --------------------------------------------------------------------------------------------
# cross-check: the exact files the Kotlin probe writes
# --------------------------------------------------------------------------------------------

def kotlin_float(x) -> str:
    """Kotlin / Java Float.toString: shortest round-trip digits, scientific below 1e-3 and from 1e7."""
    f = np.float32(x)
    if np.isnan(f):
        return "NaN"
    if np.isinf(f):
        return "Infinity" if f > 0 else "-Infinity"
    if f == 0:
        return "-0.0" if np.signbit(f) else "0.0"
    if 1e-3 <= abs(float(f)) < 1e7:
        return np.format_float_positional(f, unique=True, trim="0")
    m, e = np.format_float_scientific(f, unique=True, trim="0", exp_digits=1).split("e")
    return f"{m}E{int(e)}"


def kotlin_json(value, depth: int = 0) -> str:
    """Capabilities.kt Json.write: maps two levels deep or more on one line."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(kotlin_json(v, depth + 1) for v in value) + "]"
    if depth >= 2 or not value:
        return "{" + ", ".join(f"{json.dumps(k)}: {kotlin_json(v, depth + 1)}" for k, v in value.items()) + "}"
    pad = "  " * (depth + 1)
    body = ",\n".join(f"{pad}{json.dumps(k)}: {kotlin_json(v, depth + 1)}" for k, v in value.items())
    return "{\n" + body + "\n" + "  " * depth + "}"


ANDROID_FLOATS = {"hr": ["v0"], "acc": ["v0", "v1", "v2"], "gyro": ["v0", "v1", "v2"]}
SDK_FLOATS = {"sdk_ecg": ["ecg_mv", "max_threshold_mv", "min_threshold_mv"]}


def _kotlin_rewrite(run: Path) -> None:
    """Rewrites a synthetic run as the app writes it: Float.toString for float sensor values, ints
    as ints, IBI lists joined with '|', no CSV for a source that delivered nothing."""
    for p in run.glob("*.csv"):
        if p.stat().st_size == 0:
            p.unlink()
    for alias, cols in {**ANDROID_FLOATS, **SDK_FLOATS}.items():
        path = run / f"{alias}.csv"
        if not path.is_file():
            continue
        lines = path.read_text().splitlines()
        head = lines[0].split(",")
        idx = [head.index(c) for c in cols]
        out = [lines[0]]
        for ln in lines[1:]:
            parts = ln.split(",")
            for i in idx:
                parts[i] = kotlin_float(float(parts[i]))
            out.append(",".join(parts))
        path.write_text("\n".join(out) + "\n")
    t0 = ws.T0_MS + 5_000
    ts = t0 + np.arange(6) * 1000
    # SamsungSdk.kt columns; SKIN_TEMPERATURE floats, SpO2 ints, status lists as written
    skin = ["timestamp_ms,wall_ms,object_c,ambient_c,status"] + [
        f"{t},{t0 + 30_000},{kotlin_float(33.4 + 0.01 * k)},{kotlin_float(27.25)},0" for k, t in enumerate(ts)]
    (run / "sdk_skin.csv").write_text("\n".join(skin) + "\n")
    spo2 = ["timestamp_ms,wall_ms,spo2,hr,status,accuracy_flag"] + [
        f"{t},{t + 400},{0 if k < 5 else 97},{64},{0 if k < 5 else 2},{1}" for k, t in enumerate(ts)]
    (run / "sdk_spo2.csv").write_text("\n".join(spo2) + "\n")
    # Kotlin writes a nanosecond-scale float like 1.0E-4 for a tiny gyro reading: make sure there is one
    gyro = run / "gyro.csv"
    lines = gyro.read_text().splitlines()
    parts = lines[1].split(",")
    parts[3] = kotlin_float(1.0e-4)
    lines[1] = ",".join(parts)
    gyro.write_text("\n".join(lines) + "\n")
    caps = ws.capabilities("all", ecg=True)
    caps["features"]["skin_temp"] = {"available": True, "tier": "all", "sources": ["sdk_skin"], "reason": ""}
    caps["features"]["spo2"] = {"available": True, "tier": "all", "sources": ["sdk_spo2"], "reason": ""}
    (run / "capabilities.json").write_text(kotlin_json(caps) + "\n")


def test_kotlin_file_layout_cross_check(base, tmp_path):
    src, truth, ref = base["ecg"]
    run = clone(src, tmp_path)
    _kotlin_rewrite(run)
    assert "E-4" in (run / "gyro.csv").read_text()
    assert json.loads((run / "capabilities.json").read_text())["features"]["ppg"]["sources"] == ["sdk_ppg", "hr_raw"]
    tables = w.load_tables(run)
    assert set(tables) == {"hr_raw", "hr", "acc", "gyro", "sdk_ppg", "sdk_hr", "sdk_ecg", "sdk_skin", "sdk_spo2"}
    expect = {"hr_raw": ["event_ns", "wall_ms", "accuracy"] + [f"v{i}" for i in range(16)],
              "hr": ["event_ns", "wall_ms", "accuracy", "v0"],
              "acc": ["event_ns", "wall_ms", "accuracy", "v0", "v1", "v2"],
              "gyro": ["event_ns", "wall_ms", "accuracy", "v0", "v1", "v2"],
              "sdk_ppg": ["timestamp_ms", "wall_ms", "green", "green_status", "ir", "ir_status", "red", "red_status"],
              "sdk_hr": ["timestamp_ms", "wall_ms", "hr", "hr_status", "ibi_ms", "ibi_status"],
              "sdk_ecg": ["timestamp_ms", "wall_ms", "ecg_mv", "lead_off", "sequence", "ppg_green",
                          "max_threshold_mv", "min_threshold_mv"],
              "sdk_skin": ["timestamp_ms", "wall_ms", "object_c", "ambient_c", "status"],
              "sdk_spo2": ["timestamp_ms", "wall_ms", "spo2", "hr", "status", "accuracy_flag"]}
    for alias, cols in expect.items():
        assert tables[alias].columns == cols, alias
    assert tables["gyro"]["v0"].dtype == pl.Float64 and tables["sdk_hr"]["ibi_ms"].dtype == pl.String

    res = w.analyze_run(run)
    assert res["capabilities_source"] == "file" and res["tier"] == "all"
    assert not {"unreadable_csv", "capabilities_unreadable", "capabilities_schema"} & codes(res)
    assert res["features_used"] == ["ppg", "hr", "ibi", "motion", "skin_temp", "ecg", "spo2"]
    assert res["metrics"]["hrv"]["chosen"]["source"] == "sdk_ecg"
    for metric in ("hr", "hrv"):
        for e in ref["metrics"][metric]["estimates"]:
            got = est(res, metric, e["source"])
            if metric == "hr":
                assert abs(got["value"] - e["value"]) <= 0.2, e["source"]
            else:
                assert abs(got["value"]["rmssd_ms"] - e["value"]["rmssd_ms"]) <= 0.5, e["source"]
            assert got["reliability"] == e["reliability"], e["source"]
    assert res["metrics"]["skin_temp"]["chosen"]["value"]["object_c_median"] == pytest.approx(33.425, abs=0.01)
    assert res["metrics"]["spo2"]["chosen"]["value"] == 97.0


def test_kotlin_float_matches_java(tmp_path):
    if shutil.which("java") is None:
        pytest.skip("no java")
    src = tmp_path / "F.java"
    src.write_text(
        "import java.io.*;\npublic class F { public static void main(String[] a) throws Exception {\n"
        "  BufferedReader r = new BufferedReader(new InputStreamReader(System.in)); String l;\n"
        "  while ((l = r.readLine()) != null) System.out.println(Float.toString(Float.intBitsToFloat(Integer.parseInt(l.trim()))));\n"
        "}}\n")
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(0, 1, 500), rng.normal(9.8, 3, 300), rng.normal(0, 0.003, 300),
                           10.0 ** rng.uniform(-8, 9, 300), [0.001, 1e7, 1.0, -0.0, np.nan]]).astype(np.float32)
    proc = subprocess.run(["java", str(src)], input="\n".join(map(str, vals.view(np.int32))), capture_output=True,
                          text=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"java failed: {proc.stderr[:200]}")
    assert proc.stdout.split() == [kotlin_float(v) for v in vals]
