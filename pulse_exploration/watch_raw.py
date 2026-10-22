"""Decode and analyse tier "25hz" recordings from the watch probe app.

`com.samsung.sensor.hr_raw` (Galaxy Watch6 Classic, One UI Watch 8) delivers 16 integer channels at
25 Hz through the plain Android SensorManager. Channel meanings below are inferred from one recording
(SNR of the cardiac peak, value ranges); they are not documented by Samsung:

- v0: status (1 off the wrist, 2 on the wrist)
- v1, v4: raw light level of two PPG channels (~2.7 M counts), pulsatile
- v5 (== v7): pulsatile AC channel, signed 22-bit with an offset of 2**21
- v6, v8: small signals around 2**21 (ambient or idle LEDs)
- v9-v14 == 2**21 and v2, v3, v15 == 0: unused

Usage: uv run python -m pulse_exploration.watch_raw data/runs/<run>
"""

import sys
from pathlib import Path

import neurokit2 as nk
import numpy as np
import polars as pl
from scipy import signal

FS = 25.0
OFFSET = 2**21
CHANNELS = {"status": "v0", "ppg_a": "v1", "ppg_b": "v4", "ppg_ac": "v5", "aux_a": "v6", "aux_b": "v8"}


# Samsung SDK list values (IBI lists) are written joined with '|'; keep them as text.
LIST_COLUMNS = frozenset({"ibi_ms", "ibi_status"})


def read_csv(path: Path) -> pl.DataFrame | None:
    """One source CSV, or None when the source delivered nothing (empty file or header only)."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open() as fh:
        header = fh.readline().strip().split(",")
    overrides = {c: pl.String for c in header if c in LIST_COLUMNS}
    df = pl.read_csv(path, infer_schema_length=None, schema_overrides=overrides)
    return df if df.height else None


def load(run: Path) -> dict[str, pl.DataFrame]:
    """Every non-empty CSV of a run, keyed by signal alias."""
    return {p.stem: df for p in sorted(Path(run).glob("*.csv")) if (df := read_csv(p)) is not None}


def decode_hr_raw(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Named channels with time in seconds from the first sample (sensor clock)."""
    out = {"t": (df["event_ns"].to_numpy() - df["event_ns"][0]) / 1e9}
    for name, col in CHANNELS.items():
        x = df[col].to_numpy().astype(float)
        out[name] = x - OFFSET if name in ("ppg_ac", "aux_a", "aux_b") else x
    return out


def pulse_signal(t: np.ndarray, x: np.ndarray, invert: bool) -> tuple[np.ndarray, np.ndarray]:
    """Uniform 25 Hz grid, 0.5-4 Hz band-pass, systolic peaks pointing up."""
    grid = np.arange(t[0], t[-1], 1 / FS)
    y = np.interp(grid, t, x)
    y = -y if invert else y
    b, a = signal.butter(3, [0.5 / (FS / 2), 4 / (FS / 2)], "band")
    return grid, signal.filtfilt(b, a, y - y.mean())


def analyse(run: Path) -> dict:
    data = load(run)
    raw = decode_hr_raw(data["hr_raw"])
    t = raw["t"]
    # Raw light level drops as blood volume rises, so invert ppg_a; pick the AC sign that matches it.
    _, a = pulse_signal(t, raw["ppg_a"], invert=True)
    _, ac = pulse_signal(t, raw["ppg_ac"], invert=False)
    ac_sign = 1.0 if np.corrcoef(a, ac)[0, 1] >= 0 else -1.0
    grid, ppg = pulse_signal(t, ac_sign * raw["ppg_ac"], invert=False)

    peaks = nk.ppg_findpeaks(ppg, sampling_rate=int(FS), method="elgendi")["PPG_Peaks"]
    ibi = np.diff(grid[peaks]) * 1000
    ibi = ibi[(ibi > 300) & (ibi < 2000)]
    f, p = signal.welch(ppg, fs=FS, nperseg=min(len(ppg), 512))
    band = (f > 0.6) & (f < 3.5)

    result = {
        "tier": "25hz",
        "duration_s": round(float(t[-1]), 1),
        "fs_measured": round(float(1 / np.median(np.diff(t))), 2),
        "on_wrist": bool(np.all(raw["status"] == 2)),
        "ac_matches_inverted_raw": ac_sign > 0,
        "hr_fft_bpm": round(float(f[band][np.argmax(p[band])] * 60), 1),
        "hr_peaks_bpm": round(float(60000 / np.median(ibi)), 1) if len(ibi) else None,
        "beats": int(len(peaks)),
        "rmssd_ms_unreliable_at_25hz": round(float(np.sqrt(np.mean(np.diff(ibi) ** 2))), 1) if len(ibi) > 2 else None,
    }
    if "hr" in data:
        watch = data["hr"]["v0"].to_numpy()
        watch = watch[watch > 0]
        result["watch_hr_median_bpm"] = float(np.median(watch)) if len(watch) else None
        result["watch_hr_range_bpm"] = [float(watch.min()), float(watch.max())] if len(watch) else None
    if "acc" in data:
        acc = data["acc"].select(["v0", "v1", "v2"]).to_numpy()
        result["motion_std_ms2"] = round(float(np.linalg.norm(acc, axis=1).std()), 3)

    plot(run, grid, ppg, peaks, raw, result)
    return result


def plot(run: Path, grid, ppg, peaks, raw, result) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(raw["t"], raw["ppg_a"], lw=0.8, label="ppg_a (v1)")
    axes[0].plot(raw["t"], raw["ppg_b"], lw=0.8, label="ppg_b (v4)")
    axes[0].set_ylabel("raw counts")
    axes[0].legend(loc="upper right")
    axes[1].plot(raw["t"], raw["ppg_ac"], lw=0.8, color="tab:green", label="ppg_ac (v5 - 2^21)")
    axes[1].set_ylabel("AC counts")
    axes[1].legend(loc="upper right")
    axes[2].plot(grid, ppg, lw=0.8, color="black")
    axes[2].plot(grid[peaks], ppg[peaks], "r.", label=f"{len(peaks)} beats")
    axes[2].set_ylabel("filtered PPG")
    axes[2].set_xlabel("seconds")
    axes[2].legend(loc="upper right")
    fig.suptitle(
        f"{run.name}: HR {result['hr_peaks_bpm']} bpm (peaks), {result['hr_fft_bpm']} (FFT), "
        f"watch {result.get('watch_hr_median_bpm')} | tier 25hz"
    )
    fig.tight_layout()
    fig.savefig(run / "ppg.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    import json

    run = Path(sys.argv[1])
    print(json.dumps(analyse(run), indent=1))
    print(f"plot: {run / 'ppg.png'}")
