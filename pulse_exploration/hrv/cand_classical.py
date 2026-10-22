"""Classical candidate: band-pass, Elgendi systolic peaks, spline sub-sample timing, Kubios, quality gate.

Pipeline, per contiguous stretch of the 25 Hz watch PPG (split where samples are > 200 ms apart):
1. Invert the green ADC (it falls as blood volume rises, so systolic peaks are ADC minima) and
   band-pass it 0.5-4 Hz (Butterworth order 2, zero-phase `sosfiltfilt`).
2. Systolic peaks: NeuroKit2 `ppg_findpeaks(method="elgendi")` on the filtered signal divided by its
   5 s running RMS. Elgendi's threshold offset is the mean squared signal of its whole input, so
   without the scaling the motion bursts of an hour-long recording hide most quiet beats.
3. Sub-sample timing on a cubic spline through the filtered samples (at their own timestamps),
   evaluated on a 1 kHz grid. The timing point (`fiducial`) is found from the systolic peak:
   "peak" spline maximum within +-1 sample of the detected peak;
   "slope" steepest upstroke in the 300 ms before the peak;
   "min" pre-upstroke minimum in the 400 ms before the steepest upstroke;
   "foot" intersecting tangents (horizontal through "min", tangent at "slope");
   "mid" (default) where the upstroke crosses half way from "min" to "peak".
   `refine="parabolic"` (fiducial "peak" only) fits a parabola through the 3 samples instead.
   Then (`align_iters` > 0) each beat is shifted, within +-`align_max_shift_ms`, to the lag that
   best correlates its upstroke (`align_ms` around the beat) with the running median of its
   `template_beats` neighbours, twice; this averages timing over the whole upstroke instead of
   trusting one interpolated point.
4. Artifacts: NeuroKit2 `signal_fixpeaks(method="Kubios")` on the beat times (1 ms resolution). Extra
   beats it removes are dropped; beats it moves or inserts keep the corrected position with
   quality 0, so the bench drops every interval that touches them.
5. Quality gate:
   - per beat: Pearson r of the beat's waveform (`template_ms` around the beat) with the running
     median of its `template_beats` neighbours; quality = 0.5 at `template_min_corr` (the bench
     keeps beats with quality >= 0.5), 1 at r = 1. Beats next to an interval outside
     `ibi_range_ms` or more than `max_local_dev` from the median of `local_median_beats`
     intervals get quality 0, as do the first and last beat of each stretch (the bench would join
     them across the gap);
   - per 10 s chunk (absolute 10 s boundaries): rejected when the accelerometer magnitude SD
     exceeds `acc_sd_max` m/s^2 or fewer than `chunk_min_ppg_frac` of its PPG samples are present.
     Beats in rejected chunks get quality 0 and `accept_windows` lists the accepted chunks.

Defaults were tuned on the development participants P02-P06 only (bench, 60 s windows; RMSSD
median |error|, bias, coverage): defaults 5.5 ms, +4.3, 0.61. Timing on the systolic peak is worse
(spline 6.7 ms, +8.1, 0.30, because few peak-aligned beats pass the template check; parabolic
6.8 ms; on the sample grid 17.1 ms, +17.9), as are "foot" (6.5, +5.6, 0.56) and "mid" without
template alignment (6.7, +6.0, 0.61). Without the local-median interval check the bias rises to
+18 ms (limits of agreement -73..+110). `template_min_corr` 0.9 and `acc_sd_max` 1.5 (between
the seated-rest 90th percentile of 10 s ACC SD, 0.7, and the walking 10th percentile, 1.9) trade
a few points of coverage for error; Kubios changes little once the other checks run (5.6 ms, +4.8
without it).

The estimator sees only the watch PPG and accelerometer (bench `Estimator` protocol).
CLI: uv run python -m pulse_exploration.hrv.bench pulse_exploration.hrv.cand_classical:ClassicalEstimator
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter

GAP_MS = 200.0  # split the stream where consecutive samples are further apart than this
MIN_SEGMENT_S = 10.0  # shorter stretches are skipped
FS_NOMINAL = 25.0  # Galaxy Watch PPG rate, for the chunk presence check
FIDUCIALS = ("peak", "slope", "min", "foot", "mid")


def _znorm(a: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm rows (last axis)."""
    a = a - a.mean(-1, keepdims=True)
    return a / (np.sqrt((a * a).sum(-1, keepdims=True)) + 1e-12)


def _parabolic(y0: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """Vertex offset (in steps, within +-0.5) of the parabola through three equally spaced points
    around a maximum; 0 where they are not concave."""
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(den < 0, 0.5 * (y0 - y2) / den, 0.0)
    return np.clip(delta, -0.5, 0.5)


class ClassicalEstimator:
    name = "classical"

    def __init__(
        self,
        band_hz: tuple[float, float] = (0.5, 4.0),
        filter_order: int = 2,
        detector: str = "elgendi",
        norm_s: float = 5.0,
        fiducial: str = "mid",
        refine: str = "spline",
        upsample_hz: float = 1000.0,
        align_iters: int = 2,
        align_ms: tuple[float, float] = (-100.0, 100.0),
        align_max_shift_ms: float = 40.0,
        kubios: bool = True,
        template_beats: int = 31,
        template_ms: tuple[float, float] = (-300.0, 400.0),
        template_min_corr: float = 0.9,
        ibi_range_ms: tuple[float, float] = (300.0, 2000.0),
        max_local_dev: float | None = 0.2,
        local_median_beats: int = 11,
        chunk_s: float = 10.0,
        acc_sd_max: float | None = 1.5,
        chunk_min_ppg_frac: float = 0.8,
        name: str | None = None,
    ):
        if fiducial not in FIDUCIALS:
            raise ValueError(f"fiducial must be one of {FIDUCIALS}, not {fiducial!r}")
        if refine not in ("spline", "parabolic", "none"):
            raise ValueError(f"refine must be spline, parabolic or none, not {refine!r}")
        if refine != "spline" and fiducial != "peak":
            raise ValueError("fiducials other than 'peak' need refine='spline'")
        self.band_hz = band_hz
        self.filter_order = filter_order
        self.detector = detector
        self.norm_s = norm_s
        self.fiducial = fiducial
        self.refine = refine
        self.upsample_hz = upsample_hz
        self.align_iters = align_iters
        self.align_ms = align_ms
        self.align_max_shift_ms = align_max_shift_ms
        self.kubios = kubios
        self.template_beats = template_beats
        self.template_ms = template_ms
        self.template_min_corr = template_min_corr
        self.ibi_range_ms = ibi_range_ms
        self.max_local_dev = max_local_dev
        self.local_median_beats = local_median_beats
        self.chunk_s = chunk_s
        self.acc_sd_max = acc_sd_max
        self.chunk_min_ppg_frac = chunk_min_ppg_frac
        if name:
            self.name = name

    # ----------------------------------------------------------------------------------------------
    # steps
    # ----------------------------------------------------------------------------------------------

    def _detect(self, y: np.ndarray, fs: float) -> np.ndarray:
        import neurokit2 as nk

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = nk.ppg_findpeaks(y, sampling_rate=int(round(fs)), method=self.detector)
        except (IndexError, ValueError):  # the detectors raise when they find no pulse wave at all
            return np.zeros(0, int)
        return np.unique(np.asarray(info["PPG_Peaks"], dtype=np.int64))

    def _refine(self, t: np.ndarray, y: np.ndarray, pk: np.ndarray, cs: CubicSpline) -> np.ndarray:
        """Fiducial time (ms) for each detected peak index (see the module docstring)."""
        n = len(y)
        step = 1000.0 / self.upsample_hz
        r = np.arange(len(pk))
        if self.refine == "none":
            return t[pk]
        if self.refine == "parabolic":
            m = np.clip(pk, 1, n - 2)
            return np.interp(m + _parabolic(y[m - 1], y[m], y[m + 1]), np.arange(n), t)
        # spline maximum within +-1 sample of the detected peak
        lo, hi = t[np.clip(pk - 1, 0, n - 1)], t[np.clip(pk + 1, 0, n - 1)]
        k = int(np.ceil(np.max(hi - lo) / step)) + 1
        g = np.minimum(lo[:, None] + step * np.arange(k)[None, :], hi[:, None])
        t_peak = g[r, np.argmax(cs(g), axis=1)]
        if self.fiducial == "peak":
            return t_peak
        # steepest upstroke in the 300 ms before the peak
        g = np.maximum(t_peak[:, None] - step * np.arange(int(300.0 / step) + 1)[None, ::-1], t[0])
        dv = cs.derivative(1)(g)
        j = np.argmax(dv, axis=1)
        t_slope, slope = g[r, j], dv[r, j]
        if self.fiducial == "slope":
            return t_slope
        # pre-upstroke minimum in the 400 ms before the steepest upstroke
        g = np.maximum(t_slope[:, None] - step * np.arange(int(400.0 / step) + 1)[None, ::-1], t[0])
        v = cs(g)
        jm = np.argmin(v, axis=1)
        t_min, y_min = g[r, jm], v[r, jm]
        if self.fiducial == "min":
            return t_min
        if self.fiducial == "foot":  # horizontal through the minimum meets the tangent at the slope
            with np.errstate(divide="ignore", invalid="ignore"):
                foot = t_slope - np.where(slope > 0, (cs(t_slope) - y_min) / slope, 0.0)
            return np.clip(foot, t_slope - 400.0, t_slope)
        # "mid": first crossing of half amplitude between the minimum and the peak
        g = t_min[:, None] + (t_peak - t_min)[:, None] * np.linspace(0, 1, 101)[None, :]
        v = cs(g)
        half = 0.5 * (y_min + cs(t_peak))
        j = np.clip(np.argmax(v >= half[:, None], axis=1), 1, 100)
        a0, a1 = v[r, j - 1], v[r, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.clip(np.where(a1 > a0, (half - a0) / (a1 - a0), 0.0), 0.0, 1.0)
        return g[r, j - 1] + frac * (g[r, j] - g[r, j - 1])

    def _template_align(self, tb: np.ndarray, cs: CubicSpline, t0: float, t1: float) -> np.ndarray:
        """Shift each beat to the lag (within +-align_max_shift_ms, 2 ms grid + parabolic vertex)
        that best correlates its waveform (align_ms around the beat) with the running median of its
        neighbours' waveforms; repeated align_iters times."""
        if len(tb) < 5:
            return tb
        off = np.arange(self.align_ms[0], self.align_ms[1] + 1e-9, 10.0)
        shifts = np.arange(-self.align_max_shift_ms, self.align_max_shift_ms + 1e-9, 2.0)
        r = np.arange(len(tb))
        for _ in range(self.align_iters):
            snip = _znorm(cs(np.clip(tb[:, None] + off[None, :], t0, t1)))
            tmpl = _znorm(median_filter(snip, size=(self.template_beats, 1), mode="nearest"))
            c = np.empty((len(tb), len(shifts)))
            for k, sh in enumerate(shifts):  # one lag at a time keeps memory at n_beats x n_points
                c[:, k] = (_znorm(cs(np.clip(tb[:, None] + sh + off[None, :], t0, t1))) * tmpl).sum(1)
            j = np.clip(np.argmax(c, axis=1), 1, len(shifts) - 2)
            tb = tb + shifts[j] + _parabolic(c[r, j - 1], c[r, j], c[r, j + 1]) * (shifts[1] - shifts[0])
        return tb

    def _template_corr(self, tb: np.ndarray, cs: CubicSpline, t0: float, t1: float) -> np.ndarray:
        """Pearson r of each beat's waveform with the running median of its neighbours (0 where
        the waveform window leaves the stretch)."""
        if len(tb) < 3:
            return np.zeros(len(tb))
        off = np.arange(self.template_ms[0], self.template_ms[1] + 1e-9, 10.0)
        g = tb[:, None] + off[None, :]
        inside = (g[:, 0] >= t0) & (g[:, -1] <= t1)
        a = _znorm(cs(np.clip(g, t0, t1)))
        b = _znorm(median_filter(a, size=(self.template_beats, 1), mode="nearest"))
        return np.where(inside, (a * b).sum(1), 0.0)

    def _kubios(self, tb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Kubios correction on beat times (ms). Returns corrected times and a flag per returned beat
        (True = moved or inserted by Kubios); unchanged beats keep their sub-sample time."""
        import neurokit2 as nk

        if len(tb) < 10:
            return tb, np.zeros(len(tb), bool)
        base = tb[0]
        raw, first = np.unique(np.round(tb - base).astype(np.int64), return_index=True)
        tb = tb[first]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, fixed = nk.signal_fixpeaks(raw, sampling_rate=1000, method="Kubios", iterative=True)
        except Exception:  # noqa: BLE001  (NeuroKit fails on pathological peak trains)
            return tb, np.zeros(len(tb), bool)
        fixed = np.unique(np.asarray(fixed, dtype=np.int64))
        pos = np.minimum(np.searchsorted(raw, fixed), len(raw) - 1)
        same = raw[pos] == fixed
        return np.where(same, tb[pos], fixed + base).astype(np.float64), ~same

    def _interval_flags(self, tb: np.ndarray) -> np.ndarray:
        """True for beats next to an interval outside ibi_range_ms or far from the local median."""
        if len(tb) < 3:
            return np.ones(len(tb), bool)
        ibi = np.diff(tb)
        badi = (ibi < self.ibi_range_ms[0]) | (ibi > self.ibi_range_ms[1])
        if self.max_local_dev is not None:
            med = median_filter(ibi, size=self.local_median_beats, mode="nearest")
            badi |= np.abs(ibi - med) > self.max_local_dev * med
        bad = np.zeros(len(tb), bool)
        bad[:-1] |= badi
        bad[1:] |= badi
        return bad

    def segment(self, t: np.ndarray, x: np.ndarray) -> dict:
        """Beats of one contiguous stretch (t ms, x raw ADC), with intermediate results: beats_ms,
        quality, template_corr, kubios_corrected, interval_flag (per beat), filtered (per sample,
        inverted), peaks (detected sample indices)."""
        fs = 1000.0 / float(np.mean(np.diff(t)))
        sos = signal.butter(self.filter_order, self.band_hz, "bandpass", fs=fs, output="sos")
        y = signal.sosfiltfilt(sos, -(x - np.median(x)))  # inverted: systolic peaks are maxima
        k = int(np.clip(self.norm_s * fs, 1, len(y)))  # np.convolve "same" returns max(len) samples
        rms = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same")) + 1e-9
        pk = self._detect(y / rms, fs)
        pk = pk[(pk > 0) & (pk < len(y) - 1)]
        out = {"filtered": y, "peaks": pk, "beats_ms": np.zeros(0), "quality": np.zeros(0), "template_corr": np.zeros(0)}
        out.update({"kubios_corrected": np.zeros(0, bool), "interval_flag": np.zeros(0, bool)})
        if len(pk) < 3:
            return out
        cs = CubicSpline(t, y)
        tb = np.sort(self._refine(t, y, pk, cs))
        if self.align_iters:
            tb = np.sort(self._template_align(tb, cs, t[0], t[-1]))
        corrected = np.zeros(len(tb), bool)
        if self.kubios:
            tb, corrected = self._kubios(tb)
        corr = self._template_corr(tb, cs, t[0], t[-1])
        c = self.template_min_corr
        q = np.clip(0.5 + (corr - c) / (2 * max(1.0 - c, 1e-6)), 0.0, 1.0)
        flag = self._interval_flags(tb)
        q[corrected | flag] = 0.0
        out.update({"beats_ms": tb, "quality": q, "template_corr": corr, "kubios_corrected": corrected, "interval_flag": flag})
        return out

    def _chunks(self, tb: np.ndarray, q: np.ndarray, ppg_t: np.ndarray, acc_t: np.ndarray, acc_xyz: np.ndarray):
        """10 s chunk gate (accelerometer magnitude SD, PPG presence): accepted chunks as
        [(start_ms, end_ms)] and the beat quality with beats in rejected chunks set to 0."""
        if not len(ppg_t):
            return [], q
        W = self.chunk_s * 1000.0
        c0 = np.floor(ppg_t[0] / W) * W
        n = int(np.ceil((ppg_t[-1] - c0) / W)) + 1
        edges = c0 + W * np.arange(n + 1)
        ok = np.histogram(ppg_t, bins=edges)[0] >= self.chunk_min_ppg_frac * self.chunk_s * FS_NOMINAL
        if self.acc_sd_max is not None and len(acc_t) > 10:
            mag = np.linalg.norm(acc_xyz, axis=1)
            idx = np.searchsorted(edges, acc_t, side="right") - 1
            inb = (idx >= 0) & (idx < n)
            cnt = np.bincount(idx[inb], minlength=n).astype(float)
            s1 = np.bincount(idx[inb], weights=mag[inb], minlength=n)
            s2 = np.bincount(idx[inb], weights=mag[inb] ** 2, minlength=n)
            with np.errstate(divide="ignore", invalid="ignore"):
                sd = np.sqrt(np.clip(np.where(cnt > 1, (s2 - s1 * s1 / cnt) / (cnt - 1), np.nan), 0, None))
            ok &= ~(sd > self.acc_sd_max)  # chunks without ACC samples are not rejected for motion
        if len(tb):
            q = np.where(ok[np.clip(np.searchsorted(edges, tb, side="right") - 1, 0, n - 1)], q, 0.0)
        return [(float(edges[i]), float(edges[i + 1])) for i in np.where(ok)[0]], q

    # ----------------------------------------------------------------------------------------------
    # bench protocol
    # ----------------------------------------------------------------------------------------------

    def estimate(self, ppg_t_ms: np.ndarray, ppg: np.ndarray, acc_t_ms: np.ndarray, acc_xyz: np.ndarray) -> dict:
        t_all = np.asarray(ppg_t_ms, dtype=np.float64)
        x_all = np.asarray(ppg, dtype=np.float64)
        cuts = np.r_[0, np.where(np.diff(t_all) > GAP_MS)[0] + 1, len(t_all)]
        beats, qual = [], []
        for a, b in itertools.pairwise(cuts):
            if b - a < 4 or t_all[b - 1] - t_all[a] < MIN_SEGMENT_S * 1000:
                continue
            seg = self.segment(t_all[a:b], x_all[a:b])
            q = seg["quality"].copy()
            if len(q):
                q[[0, -1]] = 0.0  # the bench would join these beats with the next stretch's
            beats.append(seg["beats_ms"])
            qual.append(q)
        tb = np.concatenate(beats) if beats else np.zeros(0)
        q = np.concatenate(qual) if qual else np.zeros(0)
        accept, q = self._chunks(tb, q, t_all, np.asarray(acc_t_ms, dtype=np.float64), np.asarray(acc_xyz, dtype=np.float64))
        return {"beats_ms": tb, "quality": q, "accept_windows": accept}
