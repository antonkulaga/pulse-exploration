"""Exploratory "PPG age" from one averaged pulse beat with AI-PPG Age (never an absolute age).

Model: AI-PPG Age (Nie, Zhao, Tang, Li, Hong; arXiv 2502.12990, Communications Medicine 2025), weights
from Hugging Face ``Ngks03/PPG-VascularAge`` (MIT, ungated) at a pinned commit, checked by SHA-256 and
loaded with ``torch.load(weights_only=True)`` into ``pulse_exploration/net1d.py`` (Net1D, 389,689
parameters). Nothing else from the repository is executed, and its remote inference API is never
called. Input (model card): one averaged beat, shape (1, 1, 100), z-scored along the samples; output:
age in years. Training data: UK Biobank field 4205, the PulseTrace PCA2 finger infrared pulse, an
averaged beat of 100 samples running foot to foot with the systolic peak near sample 20 (paper Fig. 6).

Error: MAE 7.57 years (r = 0.49) on the authors' UK Biobank hold-out; zero-shot on VitalDB finger PPG
(Brag, arXiv 2603.14030) MAE 11.50 years with predictions collapsed into 38-67 years. Its outcome
evidence (a gap above 9 years: MACCE HR 2.37) holds for UK Biobank finger recordings, not for this
watch. Every value therefore carries its error band, a warning that it is not an absolute age, the
domain mismatch (here: wrist green PPG at 100 Hz, or a phone fingertip camera) and the number of beats
averaged. It is meant only as a within-person trend over repeated recordings at a fixed posture.

Average beat (built on watch_session's PPG beats: ``_ppg_input``, ``_ppg_beats``, ``_ppg_intervals``):
each accepted pulse interval (both beats template-matched, plausible, near the local median, wholly in
motion-free accelerometer windows) is cut from one beat fiducial to the next on the 0.5-12 Hz pulse,
time-normalised to 100 samples, linearly detrended (a cycle ends where it began) and z-scored. The
cycles are averaged, the average is rotated to start at the foot (the minimum in the 35 % of a cycle
before the steepest upstroke), cycles matching it worse than ``BEAT_CORR`` are dropped and the rest
averaged again. A bootstrap over the cycles gives the spread from beat sampling within one recording.

Reliability is "experimental" at best: "unreliable" without an accelerometer (watch), with fewer than
``MIN_BEATS`` motion-free beats (the average then uses beats recorded during motion), with beats that
match their average poorly, or when the average beat is a near-sinusoid with no shape left to read.

Use ``watch_estimate`` (watch_session's optional ``ppg_age`` metric for ``sdk_ppg100``),
``camera_estimate`` (fingertip traces from pulse_exploration/camera_api.py) or the CLI:

    uv run python -m pulse_exploration.ppg_age data/runs/<run> [data/camera/<id>-finger-trace ...] [--json]
    uv run python -m pulse_exploration.ppg_age --download   # fetch the weights only

Run analysis never downloads: without the weights on this machine the metric is skipped with a reason.
Research prototype, not a medical device.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

from pulse_exploration import watch_session as w

HF_REPO = "Ngks03/PPG-VascularAge"
HF_REVISION = "611d2e206c5f4838f443f0891e40445a4985ea3e"  # main on 2025-07-31
WEIGHTS_FILE = "model.pth"
WEIGHTS_SHA256 = "14589db19a2daf47c1afb128661e93391f809f02e1d507657ffaeb2163e23d52"
CONFIG = {"in_channels": 1, "base_filters": 24, "ratio": 1.0, "filter_list": [24, 48, 96, 192],
          "m_blocks_list": [2, 2, 2, 2], "kernel_size": 13, "stride": 1, "groups_width": 12,
          "n_classes": 1}  # config.json at HF_REVISION
N_PARAMS = 389_689
N_SAMPLES = 100
MAE_HOLDOUT_Y = 7.57  # UK Biobank hold-out, r = 0.49 (arXiv 2502.12990v4, Table 2)
MAE_ZERO_SHOT_Y = 11.50  # VitalDB, zero-shot (arXiv 2603.14030)
ZERO_SHOT_RANGE_Y = (38, 67)  # where the zero-shot predictions collapsed, whatever the true age
GAP_RISK_Y = 9.0  # gap above which UK Biobank participants had MACCE HR 2.37
MORPH_BAND = (0.5, 12.0)  # Hz: keeps the diastolic wave; wrist green has > 99.9 % of its pulse power < 10 Hz
FOOT_SEARCH = 0.35  # share of a cycle before the steepest upstroke searched for the foot
MIN_BEATS = 10  # the PulseTrace PCA2 averages a 10-15 s recording
# An average beat this close to a sinusoid (its first harmonic carries >= 90 % of its variance) has no
# pulse shape left for the model: a pure sinusoid reads 60-67 y depending on its phase, UK-Biobank-like
# synthetic beats (sharp upstroke, diastolic wave) correlate 0.75-0.84, the motion-ridden 2026-10-22 run 0.97
SHAPELESS_CORR = 0.95
N_BOOT = 200
WATCH_DOMAIN = "wrist green PPG at 100 Hz (Samsung PPG_ON_DEMAND, Galaxy Watch)"
TREND_NOTE = ("compare only within a person: repeated `spot ppg100` checks at a fixed posture, about 10 per "
              "quarter, reported as a trend")

_MODEL = None
_LOCK = threading.Lock()


class WeightsMissing(FileNotFoundError):
    """The AI-PPG Age weights are not in the local Hugging Face cache."""


def warning_text(domain: str) -> str:
    return (f"Exploratory AI-PPG Age, not an absolute or biological age: its error is {MAE_HOLDOUT_Y:.1f} years "
            f"(MAE) on the authors' UK Biobank hold-out and {MAE_ZERO_SHOT_Y:.1f} years zero-shot on VitalDB, "
            f"where predictions collapsed into {ZERO_SHOT_RANGE_Y[0]}-{ZERO_SHOT_RANGE_Y[1]} years. It was trained "
            f"on UK Biobank finger infrared PPG; this is {domain}. Use it only as a within-person trend.")


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------

def weights_path(download: bool = False) -> Path:
    """model.pth at HF_REVISION from the Hugging Face cache (fetched only when ``download``), checked
    by SHA-256. The repository is public: no token is sent."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError

    try:
        p = Path(hf_hub_download(HF_REPO, WEIGHTS_FILE, revision=HF_REVISION, token=False,
                                 local_files_only=not download))
    except LocalEntryNotFoundError as exc:
        raise WeightsMissing("AI-PPG Age weights are not on this machine: run "
                             "`uv run python -m pulse_exploration.ppg_age --download`") from exc
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise PermissionError(f"{HF_REPO} now needs a Hugging Face login: run `hf auth login` yourself, "
                              "then retry") from exc
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"{p}: SHA-256 {digest}, expected {WEIGHTS_SHA256}")
    return p


def load_model(download: bool = False):
    """The Net1D model in eval mode (cached; CPU)."""
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            import torch

            from pulse_exploration.net1d import Net1D

            model = Net1D(**CONFIG)
            model.load_state_dict(torch.load(weights_path(download), map_location="cpu", weights_only=True),
                                  strict=True)
            _MODEL = model.eval()
        return _MODEL


def predict_ages(beats: np.ndarray) -> np.ndarray:
    """AI-PPG Age (years) for z-scored beats of shape (n, N_SAMPLES)."""
    import torch

    model = load_model()
    x = torch.from_numpy(np.ascontiguousarray(beats, dtype=np.float32)[:, None, :])
    with torch.no_grad():
        return model(x).numpy()[:, 0].astype(float)


# --------------------------------------------------------------------------------------------
# average beat
# --------------------------------------------------------------------------------------------

def _z(x: np.ndarray) -> np.ndarray:
    """Z-score along the last axis, as the model card's ``normalize_ppg``."""
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def _cycles(grid: np.ndarray, y: np.ndarray, a: np.ndarray, e: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Each [a, e] span of the pulse, time-normalised to n samples, detrended and z-scored."""
    frac = np.arange(n + 1) / n
    c = CubicSpline(grid, y)(a[:, None] + (e - a)[:, None] * frac)
    c = c - (c[:, :1] + (c[:, -1:] - c[:, :1]) * frac)
    return _z(c[:, :n])


def _foot(m: np.ndarray) -> int:
    """Index of the foot of a periodic average beat: the minimum shortly before the steepest upstroke."""
    n = len(m)
    up = int(np.argmax(np.gradient(np.r_[m[-1], m, m[0]])[1:-1]))
    back = (up - np.arange(int(FOOT_SEARCH * n) + 1)) % n
    return int(back[np.argmin(m[back])])


def average_beat(grid: np.ndarray, y: np.ndarray, a: np.ndarray, e: np.ndarray, n: int = N_SAMPLES) -> dict | None:
    """Average beat (z-scored, starting at the foot) of the pulse ``y`` on ``grid`` over the beat-to-beat
    spans [a, e]; per-cycle correlation with it; cycles below BEAT_CORR are left out."""
    if len(a) < 3:
        return None
    cycles = _cycles(grid, y, np.asarray(a, float), np.asarray(e, float), n)
    keep = np.ones(len(cycles), bool)
    for _ in range(2):  # average, align at the foot, drop poor matches, average again
        mean = _z(cycles[keep].mean(axis=0))
        shift = _foot(mean)
        cycles, mean = np.roll(cycles, -shift, axis=1), np.roll(mean, -shift)
        corr = cycles @ mean / n
        new = corr >= w.BEAT_CORR
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    keep &= corr >= w.BEAT_CORR
    if keep.sum() < 3:
        return None
    mean = _z(cycles[keep].mean(axis=0))
    shift = _foot(mean)
    cycles, mean = np.roll(cycles, -shift, axis=1), np.roll(mean, -shift)
    return {"beat": mean, "cycles": cycles[keep], "corr": cycles @ mean / n, "kept": keep}


def sinusoid_corr(beat: np.ndarray) -> float:
    """Best correlation of a periodic beat with a sinusoid of its period (any phase)."""
    b = _z(np.asarray(beat, float))
    return float(np.sqrt(2) * np.abs(np.fft.rfft(b)[1]) / len(b))


def _bootstrap_sd(cycles: np.ndarray, predict: Callable[[np.ndarray], np.ndarray]) -> float | None:
    """Spread of the prediction over beat resamples: repeatability within this recording only."""
    if len(cycles) < 3:
        return None
    idx = np.random.default_rng(0).integers(0, len(cycles), (N_BOOT, len(cycles)))
    return float(np.std(predict(_z(cycles[idx].mean(axis=1))), ddof=1))


def _morphology(b: w._Beats, t: np.ndarray, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The pulse on the beats' uniform grid, band-passed so that the diastolic wave survives."""
    m = np.isfinite(t) & np.isfinite(x) & valid
    order = np.argsort(t[m], kind="stable")
    tv, xv = t[m][order], x[m][order]
    keep = np.r_[True, np.diff(tv) > 0]
    xg = np.interp(b.grid, tv[keep], xv[keep])
    return w._bandpass(xg - xg.mean(), b.fs, *MORPH_BAND)


def _from_beats(b: w._Beats, t: np.ndarray, x: np.ndarray, valid: np.ndarray, motion: w._Motion,
                source: str, feature: str, domain: str, input_notes: list[str],
                predict: Callable[[np.ndarray], np.ndarray] | None,
                motion_note: str | None = None) -> tuple[dict | None, str | None]:
    predict = predict or predict_ages
    iv = w._ppg_intervals(b, motion)
    from_still = int(iv["ok"].sum()) >= MIN_BEATS
    use = iv["ok"] if from_still else iv["quality"]
    if use.sum() < 3:
        return None, f"only {int(use.sum())} template-matched beats ({int(iv['ok'].sum())} motion-free)"
    avg = average_beat(b.grid, _morphology(b, t, x, valid), iv["a"][use], iv["e"][use])
    if avg is None:
        return None, "fewer than 3 beats match their average"
    n = int(avg["kept"].sum())
    try:
        age = float(predict(avg["beat"][None, :])[0])
    except WeightsMissing as exc:
        return None, str(exc)
    sd = _bootstrap_sd(avg["cycles"], predict)
    med_corr = float(np.median(avg["corr"][avg["kept"]]))
    sin_corr = sinusoid_corr(avg["beat"])
    reasons = []
    if not motion.checked and motion_note is None:
        reasons.append("no accelerometer: motion not checked")
    if not from_still:
        reasons.append(f"only {int(iv['ok'].sum())} motion-free intervals (needs {MIN_BEATS}): the average uses "
                       "beats recorded during motion")
    if n < MIN_BEATS:
        reasons.append(f"only {n} beats averaged (needs {MIN_BEATS})")
    if med_corr < w.GATE_CORR:
        reasons.append(f"median beat-to-average correlation {med_corr:.2f} (needs {w.GATE_CORR})")
    if sin_corr >= SHAPELESS_CORR:
        reasons.append(f"the average beat is a near-sinusoid (r = {sin_corr:.2f}, limit {SHAPELESS_CORR}): no pulse "
                       "shape for the model to read; a pure sinusoid reads 60-67 y")
    rel ="unreliable" if reasons else "experimental"
    warning = warning_text(domain)
    value = {"ppg_age_y": w._r(age, 1),
             "band_y": [w._r(age - MAE_HOLDOUT_Y, 1), w._r(age + MAE_HOLDOUT_Y, 1)],
             "band_zero_shot_y": [w._r(age - MAE_ZERO_SHOT_Y, 1), w._r(age + MAE_ZERO_SHOT_Y, 1)],
             "beats_averaged": n, "resample_sd_y": w._r(sd, 2), "absolute_age": False, "warning": warning}
    notes = input_notes + [
        (f"AI-PPG Age (Nie et al. 2025; Hugging Face {HF_REPO} @ {HF_REVISION[:7]}, MIT): Net1D CNN on one "
         f"averaged beat of {N_SAMPLES} samples, z-scored"),
        (f"{n} beats averaged: {'motion-free, ' if from_still else ''}template-matched pulse intervals cut beat to "
         f"beat on the {MORPH_BAND[0]}-{MORPH_BAND[1]:.0f} Hz pulse, time-normalised to {N_SAMPLES} samples, "
         "detrended, z-scored, averaged and started at the foot like UK Biobank's averaged pulse"),
        warning,
        f"domain mismatch: trained on UK Biobank finger infrared PPG (PulseTrace PCA2); input is {domain}",
        (f"error band = PPG age +/- {MAE_HOLDOUT_Y} y (UK Biobank hold-out MAE); zero-shot elsewhere "
         f"+/- {MAE_ZERO_SHOT_Y} y; resample_sd_y is beat-sampling spread within this recording, not accuracy"),
        (f"the outcome evidence (gap above {GAP_RISK_Y:.0f} y: MACCE HR 2.37) is for UK Biobank finger recordings; "
         "it does not carry over to this input"),
        TREND_NOTE,
    ]
    if motion_note:
        notes.append(motion_note)
    if reasons:
        notes.append("quality gate failed: " + "; ".join(reasons))
    details = {"gate": {"passed": not reasons, "reasons": reasons, "motion_checked": motion.checked,
                        "motion_free": from_still, "min_beats": MIN_BEATS,
                        "intervals_motion_free": int(iv["ok"].sum()), "intervals_template_matched":
                        int(iv["quality"].sum()), "beats_detected": b.n_detected,
                        "beats_dropped_vs_average": int((~avg["kept"]).sum()),
                        "median_corr_to_average": w._r(med_corr, 3), "sinusoid_corr": w._r(sin_corr, 3)},
               "beat": [w._r(v, 4) for v in avg["beat"]], "spans_s": w._merge_spans(iv["a"][use], iv["e"][use]),
               "fs_hz": w._r(b.fs), "domain": domain, "trained_on": "UK Biobank finger infrared PPG (PulseTrace PCA2)",
               "model": {"repo": HF_REPO, "revision": HF_REVISION, "sha256": WEIGHTS_SHA256, "params": N_PARAMS,
                         "mae_holdout_y": MAE_HOLDOUT_Y, "mae_zero_shot_y": MAE_ZERO_SHOT_Y,
                         "zero_shot_range_y": list(ZERO_SHOT_RANGE_Y)}}
    return w._estimate(value, source, feature, rel, n, notes, details), None


# --------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------

def watch_estimate(alias: str, df: pl.DataFrame, clock: w._Clock, motion: w._Motion,
                   predict: Callable[[np.ndarray], np.ndarray] | None = None) -> tuple[dict | None, str | None]:
    """PPG age from one watch PPG table (meant for sdk_ppg100), or the reason there is none."""
    t, x, valid, notes = w._ppg_input(alias, df, clock)
    beats = w._ppg_beats(alias, t, x, valid, motion)
    if beats is None:
        return None, "no usable PPG (under 10 s on-wrist or under 6 beats)"
    domain = WATCH_DOMAIN if alias == "sdk_ppg100" else f"wrist PPG ({alias}, {beats.fs:.0f} Hz)"
    return _from_beats(beats, t, x, valid, motion, alias, "ppg100" if alias == "sdk_ppg100" else "ppg",
                       domain, notes, predict)


def run_estimate(run_dir: str | Path, predict=None) -> tuple[dict | None, str | None]:
    """PPG age of a watch run directory (from sdk_ppg100)."""
    run = Path(run_dir)
    tables = w.load_tables(run)
    if "sdk_ppg100" not in tables:
        return None, "no sdk_ppg100 in this run (record one with `spot ppg100`)"
    clock = w._Clock(w._read_kv(run / "meta.txt"), tables)
    return watch_estimate("sdk_ppg100", tables["sdk_ppg100"], clock, w._motion(tables, clock), predict)


def camera_estimate(trace: dict, result: dict | None = None,
                    predict: Callable[[np.ndarray], np.ndarray] | None = None) -> tuple[dict | None, str | None]:
    """PPG age from a fingertip camera trace (``trace.json`` of camera_api: mode, t_ms, r, g, b). The
    channel is camera.py's primary method (inverted red, green when red is clipped); ``result`` is the
    saved ``result.json`` when there is one."""
    from pulse_exploration import camera

    if trace.get("mode") != "finger":
        return None, "PPG age needs a fingertip (mode finger) trace"
    t = np.asarray(trace["t_ms"], float) / 1000
    rgb = {k: np.asarray(trace[k], float) for k in "rgb"}
    if not result or "primary_method" not in result:
        result = camera.analyze_trace(t, rgb["r"], rgb["g"], rgb["b"], "finger", "capture")
    ch = "g" if result.get("primary_method") == "green" else "r"
    x = -rgb[ch] / max(float(np.nanmean(rgb[ch])), 1e-9)  # inverted: more blood, less light
    motion = w._Motion(None)
    beats = w._ppg_beats(f"camera_{ch}", t, x, np.isfinite(x), motion)
    if beats is None:
        return None, "no usable fingertip PPG (under 10 s or under 6 beats)"
    domain = f"phone fingertip camera PPG ({beats.fs:.0f} fps, torch, {'green' if ch == 'g' else 'red'} channel)"
    est, why = _from_beats(beats, t, x, np.isfinite(x), motion, "camera_finger", "camera", domain,
                           [(f"fingertip camera, inverted {'green' if ch == 'g' else 'red'} channel "
                             f"({result.get('primary_reason', 'camera.py primary method')})")], predict,
                           motion_note="no accelerometer: movement shows only as beats that do not match their "
                                       "average")
    if est is not None:
        est["tier"] = result.get("tier", "capture")
    return est, why


def trace_estimate(trace_dir: str | Path, predict=None) -> tuple[dict | None, str | None]:
    d = Path(trace_dir)
    trace = json.loads((d / "trace.json").read_text())
    result = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() else None
    return camera_estimate(trace, result, predict)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    paths = [a for a in argv if not a.startswith("--")]
    if "--download" in argv:
        print(f"weights: {weights_path(download=True)}")
    if not paths:
        if "--download" in argv:
            return 0
        print("usage: python -m pulse_exploration.ppg_age <run_dir | camera trace dir> ... [--json] [--download]",
              file=sys.stderr)
        return 2
    load_model(download=True)  # the CLI is an explicit request: fetch the weights once if needed
    out = []
    for p in map(Path, paths):
        est, why = trace_estimate(p) if (p / "trace.json").is_file() else run_estimate(p)
        out.append({"input": str(p), "estimate": est, "reason": why})
        if "--json" in argv:
            continue
        if est is None:
            print(f"{p}: no PPG age ({why})")
            continue
        v, g = est["value"], est["details"]["gate"]
        print(f"{p}: PPG age {v['ppg_age_y']} y, band {v['band_y'][0]}-{v['band_y'][1]} y "
              f"(zero-shot {v['band_zero_shot_y'][0]}-{v['band_zero_shot_y'][1]} y), {v['beats_averaged']} beats "
              f"averaged, beat-resample SD {v['resample_sd_y']} y, reliability {est['reliability']}")
        for r in g["reasons"]:
            print(f"  gate: {r}")
        print(f"  {v['warning']}")
    if "--json" in argv:
        print(json.dumps(w._jsonable(out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
