"""Synthetic watch run directories (no recordings of real people) following the SHARED CONTRACT.

``make_run`` writes what the watch app would upload for one run: ``capabilities.json``, one CSV
per source alias in the exact column layout of ``Sources.kt``, ``record.tsv`` and ``meta.txt``.
Every signal is generated from one set of known beat times, returned as ``WatchTruth``:

* R-peak times with respiratory sinus arrhythmia plus random beat-to-beat variation
  (RMSSD ~40 ms at 65 bpm), PPG pulse peaks a pulse transit time (~250 ms, 3 ms jitter) later.
* hr_raw: 25 Hz (24.9 Hz like the SM-R965F), 16 integer channels; v1/v4 raw light level (drops at
  systole), v5 == v7 AC channel with the 2^21 offset (follows the raw light level), v0 status
  (first second off-wrist), baseline wander, sensor noise.
* hr (1 Hz Samsung HR via Android), acc / gyro at ~100 Hz, motion bursts where requested.
  During motion every PPG channel gets large artefacts, so HRV from motion windows would be wrong.
* tier "all" adds sdk_ppg (25 Hz green / IR / red), sdk_hr (1 Hz rows with IBI lists, one invalid
  IBI with status -1 and one missing row) and optionally a 500 Hz sdk_ecg with R-peaks at the true
  beat times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

T0_MS = 1_792_700_000_000  # wall clock of the run start
BOOT_S = 450_000.0  # elapsedRealtime at the run start (event_ns clock)
OFFSET = 2**21
HR_RAW_FS = 24.9
ACC_FS = 99.6
SUPPORTED = [  # HealthTrackerType names reported by SM-R965F, Health Platform 1.7
    "PPG_GREEN", "ACCELEROMETER", "ACCELEROMETER_CONTINUOUS", "ECG", "ECG_ON_DEMAND", "PPG_IR", "PPG_RED",
    "SPO2", "SPO2_ON_DEMAND", "HEART_RATE", "HEART_RATE_CONTINUOUS", "BIA", "BIA_ON_DEMAND", "SWEAT_LOSS",
    "SKIN_TEMPERATURE", "SKIN_TEMPERATURE_ON_DEMAND", "SKIN_TEMPERATURE_CONTINUOUS", "PPG_ON_DEMAND",
    "PPG_CONTINUOUS",
]


@dataclass
class WatchTruth:
    r_peaks_s: np.ndarray
    pulse_s: np.ndarray
    duration_s: float
    motion_s: list[tuple[float, float]]
    ecg_s: tuple[float, float] | None = None
    extra: dict = field(default_factory=dict)

    def intervals(self, a: float, b: float, pulse: bool = False) -> np.ndarray:
        """NN intervals (ms) lying wholly inside [a, b]."""
        t = self.pulse_s if pulse else self.r_peaks_s
        m = (t[:-1] >= a) & (t[1:] <= b)
        return np.diff(t)[m] * 1000

    def rmssd(self, a: float, b: float, pulse: bool = False) -> float:
        return float(np.sqrt(np.mean(np.diff(self.intervals(a, b, pulse)) ** 2)))

    def hr_bpm(self, a: float, b: float) -> float:
        return float(60000 / np.median(self.intervals(a, b)))

    def in_motion(self, t: np.ndarray, pad: float = 0.0) -> np.ndarray:
        t = np.asarray(t)
        m = np.zeros(t.shape, bool)
        for a, b in self.motion_s:
            m |= (t >= a - pad) & (t <= b + pad)
        return m


def beat_times(duration_s: float, hr_bpm: float, rng: np.random.Generator, rsa_ms: float = 30.0,
               noise_ms: float = 20.0, resp_hz: float = 0.25) -> np.ndarray:
    """R-peak times (s) from -1 s to duration + 2 s."""
    mean = 60000 / hr_bpm
    t, out = -1.0 - rng.uniform(0, mean / 1000), []
    while t < duration_s + 2:
        out.append(t)
        rr = mean + rsa_ms * np.sin(2 * np.pi * resp_hz * t) + noise_ms * rng.standard_normal()
        t += max(rr, 350) / 1000
    return np.asarray(out)


def ppg_wave(ts: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Pulse waveform (systolic peak ~1 at each pulse peak, fast rise, slow fall, diastolic wave)."""
    y = np.zeros_like(ts)
    for p in peaks:
        lo, hi = np.searchsorted(ts, [p - 0.6, p + 1.2])
        tau = ts[lo:hi] - p
        sig = np.where(tau < 0, 0.07, 0.15)
        y[lo:hi] += np.exp(-0.5 * (tau / sig) ** 2) + 0.35 * np.exp(-0.5 * ((tau - 0.32) / 0.09) ** 2)
    return y


def ecg_wave(ts: np.ndarray, r_peaks: np.ndarray) -> np.ndarray:
    """PQRST complexes (mV) at the given R-peak times."""
    waves = ((-0.20, 0.12, 0.025), (-0.035, -0.12, 0.008), (0.0, 1.1, 0.010), (0.035, -0.25, 0.010),
             (0.26, 0.30, 0.045))
    y = np.zeros_like(ts)
    for r in r_peaks:
        lo, hi = np.searchsorted(ts, [r - 0.4, r + 0.6])
        tau = ts[lo:hi] - r
        for mu, amp, sd in waves:
            y[lo:hi] += amp * np.exp(-0.5 * ((tau - mu) / sd) ** 2)
    return y


def _artefact(ts: np.ndarray, truth: WatchTruth, rng: np.random.Generator) -> np.ndarray:
    """Motion artefact (units of pulse amplitude): random 0.5-3 Hz components plus jumps."""
    m = truth.in_motion(ts).astype(float)
    y = np.zeros_like(ts)
    for _ in range(8):
        y += rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * ts + rng.uniform(0, 2 * np.pi))
    y += np.cumsum(rng.standard_normal(len(ts))) * 0.15
    return 1.5 * y * m


def _sample_times(duration_s: float, fs: float, rng: np.random.Generator, start: float = 0.0,
                  jitter_s: float = 0.0005) -> np.ndarray:
    t = start + np.arange(int(duration_s * fs)) / fs
    return t + jitter_s * rng.standard_normal(len(t))


def _android_clock(t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    event_ns = np.round((BOOT_S + t) * 1e9).astype(np.int64)
    wall_ms = np.round(T0_MS + t * 1000 + rng.uniform(1, 6, len(t))).astype(np.int64)
    return event_ns, wall_ms


def _write_csv(path: Path, header: list[str], cols: list) -> int:
    n = len(cols[0])
    with path.open("w") as fh:
        fh.write(",".join(header) + "\n")
        for i in range(n):
            fh.write(",".join(_fmt(c[i]) for c in cols) + "\n")
    return n


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    f = float(v)
    return repr(round(f, 7)) if not f.is_integer() else f"{f:.1f}"


def _ints(x) -> list[int]:
    return [int(v) for v in np.round(x)]


def capabilities(tier: str, ecg: bool) -> dict:
    """capabilities.json as the watch app writes it for these synthetic runs."""
    dev = "needs Samsung Health Platform developer mode"
    if tier == "25hz":
        sdk = {"installed": True, "connected": True, "policy": "denied", "error": "SDK_POLICY_ERROR",
               "version": "1.4.01.00", "supported": SUPPORTED}
        feats = {
            "ppg": {"available": True, "tier": "25hz", "sources": ["hr_raw"], "reason": ""},
            "hr": {"available": True, "tier": "25hz", "sources": ["hr"], "reason": ""},
            "motion": {"available": True, "tier": "25hz", "sources": ["acc", "gyro"], "reason": ""},
        }
        for f in ("ibi", "skin_temp", "ppg100", "ecg", "spo2"):
            feats[f] = {"available": False, "tier": None, "sources": [], "reason": dev}
    else:
        sdk = {"installed": True, "connected": True, "policy": "allowed", "error": "", "version": "1.4.01.00",
               "supported": SUPPORTED}
        not_rec = {"available": False, "tier": None, "sources": [], "reason": "not recorded in this run"}
        feats = {
            "ppg": {"available": True, "tier": "all", "sources": ["sdk_ppg", "hr_raw"], "reason": ""},
            "hr": {"available": True, "tier": "all", "sources": ["sdk_hr", "hr"], "reason": ""},
            "ibi": {"available": True, "tier": "all", "sources": ["sdk_hr"], "reason": ""},
            "motion": {"available": True, "tier": "25hz", "sources": ["acc", "gyro"], "reason": ""},
            "skin_temp": dict(not_rec), "ppg100": dict(not_rec), "spo2": dict(not_rec),
            "ecg": ({"available": True, "tier": "all", "sources": ["sdk_ecg"], "reason": ""} if ecg
                    else dict(not_rec)),
        }
    return {"schema": 1, "tier": tier, "sdk": sdk,
            "android": {"hr_raw": True, "hr": True, "acc": True, "gyro": True}, "features": feats}


def make_run(root: Path, name: str, tier: str = "25hz", duration_s: float = 120.0, hr_bpm: float = 65.0,
             motion: list[tuple[float, float]] | None = None, ecg_s: tuple[float, float] | None = None,
             with_capabilities: bool = True, seed: int = 0) -> tuple[Path, WatchTruth]:
    rng = np.random.default_rng(seed)
    run = Path(root) / name
    run.mkdir(parents=True, exist_ok=True)
    r = beat_times(duration_s, hr_bpm, rng)
    pulse = r + 0.25 + 0.003 * rng.standard_normal(len(r))
    truth = WatchTruth(r, pulse, duration_s, list(motion or []), ecg_s)
    rows: list[list] = []  # record.tsv

    def summary(tier_label, alias, n, fs, nvals, first, error=""):
        rows.append([tier_label, alias, "true", "true", n, f"{fs:.2f}" if n > 1 else "", nvals if n else 0,
                     first if n else "", error])

    # ---- hr_raw (25 Hz) --------------------------------------------------------------------
    t = _sample_times(duration_s, HR_RAW_FS, rng)
    w = ppg_wave(t, pulse)
    w = w - w.mean()
    art = _artefact(t, truth, rng)
    breath = 0.4 * np.sin(2 * np.pi * 0.25 * t) + 0.8 * np.sin(2 * np.pi * t / 47 + 1)
    on = t >= 1.0  # first second off the wrist
    v0 = np.where(on, 2, 1)

    def light(base, gain, noise):
        x = base - gain * (w + breath + art) + noise * rng.standard_normal(len(t))
        return np.where(on, x, OFFSET)

    v1 = light(2_720_000, 4000, 150)
    v4 = light(2_680_000, 4500, 150)
    v5 = np.where(on, OFFSET - 40000 * (w + 0.5 * breath + art) + 600 * rng.standard_normal(len(t)), OFFSET)
    def small() -> np.ndarray:  # v6, v8: ambient / idle channels near 2^21
        return np.where(on, OFFSET + 300 + 20 * rng.standard_normal(len(t)), OFFSET)

    v6, v8 = small(), small()
    full, zero = np.full(len(t), OFFSET), np.zeros(len(t))
    cols = [v0, v1, zero, zero, v4, v5, v6, v5, v8] + [full] * 6 + [zero]
    ens, wall = _android_clock(t, rng)
    acc3 = np.full(len(t), 3)
    n = _write_csv(run / "hr_raw.csv", ["event_ns", "wall_ms", "accuracy"] + [f"v{i}" for i in range(16)],
                   [ens, wall, acc3] + [_ints(c) for c in cols])
    summary("25hz", "hr_raw", n, HR_RAW_FS, 16, ",".join(str(int(round(c[0]))) for c in cols))

    # ---- hr (1 Hz) ---------------------------------------------------------------------------
    th = np.arange(1.0, duration_s, 1.004)
    hr_vals = np.array([60000 / np.mean(np.diff(r[(r > x - 4) & (r <= x)]) * 1000) if ((r > x - 4) & (r <= x)).sum() > 1
                        else hr_bpm for x in th])
    hr_vals = np.round(hr_vals)
    hr_acc = np.full(len(th), 3)
    hr_vals[0], hr_acc[0] = 0.0, -1
    ens, wall = _android_clock(th, rng)
    n = _write_csv(run / "hr.csv", ["event_ns", "wall_ms", "accuracy", "v0"],
                   [ens, wall, hr_acc, hr_vals.astype(float)])
    summary("25hz", "hr", n, 1.0, 1, "0.0")

    # ---- acc / gyro (100 Hz) -------------------------------------------------------------------
    ta = _sample_times(duration_s, ACC_FS, rng)
    mm = truth.in_motion(ta)[:, None]
    g = np.array([5.62, 3.49, 4.19])
    g = g / np.linalg.norm(g) * 9.80665
    phase = rng.uniform(0, 2 * np.pi, 3)
    shake = 2.0 * np.sin(2 * np.pi * 1.8 * ta[:, None] + phase) + 0.8 * rng.standard_normal((len(ta), 3))
    acc = g + 0.03 * rng.standard_normal((len(ta), 3)) + shake * mm
    gyro = 0.003 * rng.standard_normal((len(ta), 3)) + 1.5 * np.sin(2 * np.pi * 1.8 * ta[:, None] + phase) * mm
    ens, wall = _android_clock(ta, rng)
    for alias, data in (("acc", acc), ("gyro", gyro)):
        n = _write_csv(run / f"{alias}.csv", ["event_ns", "wall_ms", "accuracy", "v0", "v1", "v2"],
                       [ens, wall, np.full(len(ta), 3), data[:, 0], data[:, 1], data[:, 2]])
        summary("25hz", alias, n, ACC_FS, 3, ",".join(f"{v:.6g}" for v in data[0]))

    if tier == "25hz":
        # the app tried the SDK trackers; the Health Platform refused them (developer mode off)
        for alias in ("sdk_ppg", "sdk_hr"):
            (run / f"{alias}.csv").write_text("")
            rows.append(["all", alias, "true", "true", 0, "", 0, "", "SDK_POLICY_ERROR"])
    else:
        # ---- sdk_ppg (25 Hz green / IR / red) -------------------------------------------------
        ts = np.arange(0.096, duration_s, 0.040)
        wp = ppg_wave(ts, pulse)
        wp = wp - wp.mean()
        ap = _artefact(ts, truth, rng)
        bp = 0.4 * np.sin(2 * np.pi * 0.25 * ts)
        ms = T0_MS + np.round(ts * 1000).astype(np.int64)
        arrival = T0_MS + (np.floor(ts / 10) * 10 + 10.05) * 1000
        green = 96000 - 3000 * (wp + bp + ap) + 60 * rng.standard_normal(len(ts))
        ir = 474000 - 9000 * (wp + bp + ap) + 150 * rng.standard_normal(len(ts))
        red = 450000 - 7000 * (wp + bp + ap) + 150 * rng.standard_normal(len(ts))
        z = np.zeros(len(ts), int)
        n = _write_csv(run / "sdk_ppg.csv", ["timestamp_ms", "wall_ms", "green", "green_status", "ir", "ir_status",
                                             "red", "red_status"],
                       [ms, _ints(arrival), _ints(green), z, _ints(ir), z, _ints(red), z])
        summary("all", "sdk_ppg", n, 25.0, 6, f"{int(green[0])},0,{int(ir[0])},0,{int(red[0])},0")

        # ---- sdk_hr (1 Hz rows, IBI lists = pulse intervals) -----------------------------------
        tr = np.arange(0.05, duration_s, 1.004)
        missing_row = int(np.argmin(np.abs(tr - 30.0)))  # a row (and its beats) the SDK never delivered
        bad_row = int(np.argmin(np.abs(tr - 50.0)))  # one IBI flagged invalid
        pi = np.diff(pulse) * 1000
        pend = pulse[1:]
        out = {k: [] for k in ("ts", "wall", "hr", "hr_status", "ibi", "ibi_status")}
        prev = -np.inf
        for k, x in enumerate(tr):
            sel = (pend > prev) & (pend <= x)
            prev = x
            if k == missing_row:
                continue
            ib = np.round(pi[sel]).astype(int).tolist()
            st = [0] * len(ib)
            if truth.in_motion(np.array([x]), pad=1.0)[0] and ib:
                ib = [int(v + rng.normal(0, 150)) for v in ib]
                st = [0 if rng.uniform() < 0.5 else -1 for _ in ib]
            if k == bad_row and ib:
                ib[0], st[0] = 1650, -1
            win = (r > x - 4) & (r <= x)
            hr_k = round(60000 / (np.mean(np.diff(r[win])) * 1000)) if win.sum() > 1 else 0
            out["ts"].append(T0_MS + int(round(x * 1000)))
            out["wall"].append(T0_MS + int(round(x * 1000)) + 500)
            out["hr"].append(hr_k if k else 0)
            out["hr_status"].append(1 if k else 0)
            out["ibi"].append("|".join(map(str, ib)))
            out["ibi_status"].append("|".join(map(str, st)))
        n = _write_csv(run / "sdk_hr.csv", ["timestamp_ms", "wall_ms", "hr", "hr_status", "ibi_ms", "ibi_status"],
                       [out["ts"], out["wall"], out["hr"], out["hr_status"], out["ibi"], out["ibi_status"]])
        summary("all", "sdk_hr", n, 1.0, 4, f"0,0,{out['ibi'][0]},{out['ibi_status'][0]}")
        truth.extra.update(missing_row_s=float(tr[missing_row]), bad_row_s=float(tr[bad_row]))

        # ---- sdk_ecg (500 Hz, finger on the key) ---------------------------------------------
        if ecg_s is not None:
            a, b = ecg_s
            te = a + np.arange(int((b - a) * 500)) / 500
            ecg = ecg_wave(te, r) + 0.015 * rng.standard_normal(len(te)) + 0.1 * np.sin(2 * np.pi * 0.3 * te)
            lead_off = np.where(te < a + 0.5, 5, 0)
            ecg = np.where(lead_off == 0, ecg, 0.0)
            # as on SM-R965F: bundles of 10 samples share one timestamp, and only a bundle's first sample
            # carries lead_off, sequence, ppg_green and the thresholds (the other nine read -1)
            head = np.arange(len(te)) % 10 == 0
            bundle_te = te[np.arange(len(te)) // 10 * 10]
            mse = T0_MS + np.round(bundle_te * 1000).astype(np.int64)
            arrival = T0_MS + (np.floor((te - a) / 2) * 2 + a + 2.1) * 1000
            n = _write_csv(run / "sdk_ecg.csv", ["timestamp_ms", "wall_ms", "ecg_mv", "lead_off", "sequence",
                                                 "ppg_green", "max_threshold_mv", "min_threshold_mv"],
                           [mse, _ints(arrival), np.round(ecg, 5), np.where(head, lead_off, -1),
                            np.where(head, np.arange(len(te)) // 10 % 256, -1), np.where(head, 85000, -1),
                            np.where(head, 2.0, -1.0), np.where(head, -2.0, -1.0)])
            summary("all", "sdk_ecg", n, 500.0, 6, f"{ecg[0]:.4f},5,0,85000,2.0,-2.0")
            truth.ecg_s = (a + 0.5, b)

    with (run / "record.tsv").open("w") as fh:
        fh.write("tier\talias\tavailable\tregistered\tevents\trate_hz\tvalues_per_event\tfirst_values\terror\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    (run / "meta.txt").write_text(
        f"model=SM-R965F\nbuild=SYNTHETIC\nsdk_int=36\nstart_wall_ms={T0_MS}\n"
        f"end_wall_ms={T0_MS + int(duration_s * 1000)}\nseconds={int(duration_s)}\n")
    if with_capabilities:
        (run / "capabilities.json").write_text(json.dumps(capabilities(tier, ecg_s is not None), indent=1))
    return run, truth
