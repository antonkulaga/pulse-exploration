"""Candidate: pulse-upstroke timing refined by template matching, 25 Hz Galaxy Watch PPG (tier "25hz").

Per contiguous PPG segment (the stream is split where samples are missing for > 200 ms):

1. Orientation and filter. The Galaxy green ADC falls as blood volume rises, so x = -ADC puts the
   systolic upstroke positive. Zero-phase Butterworth band-pass (`band_hz`).
2. Coarse beats on the 25 Hz grid: maxima of the first derivative (steepest upstroke), scaled by a
   10 s running RMS, with a refractory distance and a height threshold relative to the running
   median of the neighbouring peak heights (drops the dicrotic bump).
3. Sub-sample fiducial on a cubic spline through the filtered samples at their timestamps,
   evaluated at 1 ms steps: the maximum of the spline derivative ("maxslope", default) or the
   intersecting-tangents foot ("tangent": the tangent at maximum slope meets the horizontal through
   the preceding minimum).
4. Matched filter. Each beat's 1 kHz spline snippet (`tmpl_win_ms` around its fiducial) is
   cross-correlated (Pearson, per lag) with the running median template of its `tmpl_beats`
   neighbours over +-`max_shift_ms`; the beat moves to the correlation peak (parabolic sub-ms
   refinement) and the template is rebuilt from the aligned snippets (`n_iter` passes). The
   quality correlation is then taken over the longer `qual_win_ms` (whole pulse) at the aligned
   position; beats below `corr_min`, or whose timing peak sits at the edge of the search, are
   rejected.
5. Motion gate. Accelerometer magnitude, band-passed 0.5-10 Hz; beats where its RMS over
   `motion_win_s` exceeds `motion_max` (m/s^2) are rejected.
6. IBI artifact correction on the surviving beats (`artifact_correct`, local median RR over
   `art_med_beats` beats): extra beats (interval < `art_short` x median) are removed, long
   intervals (> `art_long` x median) get a quality-0 placeholder, and misplaced beats (a beat
   whose distance from the midpoint of its neighbours is a local maximum and exceeds `art_ddev`
   robust SDs) are rejected. Rejected beats stay in the output with quality 0, so the intervals
   touching them are dropped instead of merged.
7. Lost-sample correction (`fit_drops`, see there). The watch silently drops about one PPG
   sample in every 150-500 while its stamps stay evenly spaced, so every beat after a lost sample
   is stamped one period (40 ms) early relative to the beats before it. A Viterbi pass finds the
   chain of losses (quasi-periodic in sample index) from the interval series alone and undoes
   them; then step 6 runs again.

Output: beats_ms (template-aligned maximum upstroke; the scorer removes a constant delay),
quality (0 for rejected beats and placeholders, else 0.5 + 0.5 x scaled correlation), and
accept_windows: stretches where, over `accept_win_s`, at least `accept_min_good` of the expected
beats (window length / median RR) survived (None disables). The estimator sees PPG and
accelerometer only.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import interpolate, ndimage, signal

FS = 25.0
GAP_MS = 200.0
MIN_SEG_S = 10.0
EDGE_MS = 1500.0  # beats this close to a segment edge sit on filter transients: rejected


@dataclass
class FootTemplateEstimator:
    name: str = "foot_template"
    invert: bool = True  # Galaxy green ADC falls with the pulse upstroke
    band_hz: tuple[float, float] = (0.5, 8.0)
    filter_order: int = 2
    min_distance_s: float = 0.30
    peak_rel_height: float = 0.35  # derivative peak >= this x running median of peak heights
    fiducial: str = "maxslope"  # or "tangent"
    tmpl_win_ms: tuple[float, float] = (-100.0, 100.0)  # timing (cross-correlation) window
    qual_win_ms: tuple[float, float] = (-150.0, 300.0)  # quality-correlation window
    tmpl_beats: int = 31
    max_shift_ms: float = 60.0
    n_iter: int = 2
    corr_min: float = 0.80
    motion_win_s: float = 2.0
    motion_max: float | None = 1.0
    art: bool = True
    art_med_beats: int = 11
    art_short: float = 0.7
    art_long: float = 1.4
    art_ddev: float = 4.0
    drops: bool = True  # find and undo the watch's silently lost samples
    drop_ms: float = 40.0
    drop_n_range: tuple[float, float] = (150.0, 520.0)
    drop_jitter: float = 25.0
    drop_pick_cost: float = 10.0
    drop_g_min: float = 0.0
    accept_win_s: float = 30.0
    accept_min_good: float | None = None
    debug: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------------------------------
    def estimate(self, ppg_t_ms: np.ndarray, ppg: np.ndarray, acc_t_ms: np.ndarray, acc_xyz: np.ndarray) -> dict:
        t = np.asarray(ppg_t_ms, dtype=np.float64)
        v = np.asarray(ppg, dtype=np.float64)
        motion = motion_level(acc_t_ms, acc_xyz, self.motion_win_s)
        cuts = np.r_[0, np.where(np.diff(t) > GAP_MS)[0] + 1, len(t)]
        beats, qual, corr = [], [], []
        n_drops = 0
        art = (self.art_med_beats, self.art_short, self.art_long, self.art_ddev)
        for a, b in itertools.pairwise(cuts):
            if b - a < MIN_SEG_S * FS:
                continue
            bt, c = self._segment(t[a:b], v[a:b])
            if len(bt) < 5:
                continue
            q = np.where(np.isfinite(c) & (c >= self.corr_min), 1.0, 0.0)
            q[(bt < t[a] + EDGE_MS) | (bt > t[b - 1] - EDGE_MS)] = 0.0
            if self.motion_max is not None and motion is not None:
                q[np.interp(bt, motion[0], motion[1]) > self.motion_max] = 0.0
            if self.art:
                bt, q, c = artifact_correct(bt, q, c, *art)
            if self.drops:
                si = np.interp(bt, t[a:b], sample_index(t[a:b]))
                shift, nd = fit_drops(
                    bt, q > 0, si, self.drop_ms, self.drop_n_range, self.drop_jitter, self.drop_g_min, pick_cost=self.drop_pick_cost
                )
                bt = bt + shift
                n_drops += nd
                if self.art:
                    bt, q, c = artifact_correct(bt, q, c, *art)
            # placeholder after the segment: no interval spans a recording gap
            beats.append(np.r_[bt, bt[-1] + 1.0])
            qual.append(np.r_[q, 0.0])
            corr.append(np.r_[c, np.nan])
        if not beats:
            return {"beats_ms": np.zeros(0), "quality": np.zeros(0), "accept_windows": []}
        bt, q, c = np.concatenate(beats), np.concatenate(qual), np.concatenate(corr)
        cc = np.clip((np.nan_to_num(c) - self.corr_min) / max(1.0 - self.corr_min, 1e-6), 0.0, 1.0)
        quality = np.where(q > 0, 0.5 + 0.5 * cc, 0.0)
        out = {"beats_ms": bt, "quality": quality, "accept_windows": None}
        if self.accept_min_good is not None:
            out["accept_windows"] = accept_windows(bt, quality >= 0.5, t, self.accept_win_s, self.accept_min_good)
        self.debug = {"corr": c, "n_drops": n_drops}
        return out

    # ------------------------------------------------------------------------------------------
    def _segment(self, t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Beat times (fiducial) and quality correlation for one contiguous segment."""
        x = -(v - np.median(v)) if self.invert else v - np.median(v)
        sos = signal.butter(self.filter_order, self.band_hz, "bandpass", fs=FS, output="sos")
        y = signal.sosfiltfilt(sos, x)
        pk = coarse_peaks(y, self.min_distance_s, self.peak_rel_height)
        pk = pk[(pk > 2) & (pk < len(y) - 3)]
        if len(pk) < 5:
            return np.zeros(0), np.zeros(0)
        cs = interpolate.CubicSpline(t, y)
        tf = fine_fiducial(cs, t, pk, self.fiducial)
        if self.n_iter > 0:
            tf, corr = template_align(cs, t, tf, self.tmpl_win_ms, self.qual_win_ms, self.tmpl_beats, self.max_shift_ms, self.n_iter)
        else:
            corr = template_corr(cs, t, tf, self.qual_win_ms, self.tmpl_beats)
        o = np.argsort(tf, kind="stable")
        return tf[o], corr[o]


# ----------------------------------------------------------------------------------------------
# building blocks (module-level so tests can exercise them)
# ----------------------------------------------------------------------------------------------


def sample_index(t: np.ndarray) -> np.ndarray:
    """Index of each delivered sample on the nominal grid (a stamp gap of ~2 periods counts 2)."""
    if len(t) < 2:
        return np.zeros(len(t))
    dt = np.diff(t)
    p = float(np.median(dt))
    return np.r_[0.0, np.cumsum(np.maximum(1, np.round(dt / p)))]


def coarse_peaks(y: np.ndarray, min_distance_s: float, rel_height: float) -> np.ndarray:
    """Sample indices of maximum-upstroke points: peaks of the derivative, running-RMS scaled."""
    d = np.gradient(y)
    k = int(10 * FS)
    rms = np.sqrt(np.convolve(d * d, np.ones(k) / k, mode="same")) + 1e-12
    pk, prop = signal.find_peaks(d / rms, distance=max(1, int(round(min_distance_s * FS))), height=0.0)
    if len(pk) < 3:
        return pk
    h = prop["peak_heights"]
    # typical main-upstroke height: running median of ~21 neighbouring peaks, then a local max
    ref = ndimage.maximum_filter1d(ndimage.median_filter(h, size=21, mode="nearest"), size=5, mode="nearest")
    return pk[h >= rel_height * ref]


def fine_fiducial(cs: interpolate.CubicSpline, t: np.ndarray, pk: np.ndarray, kind: str) -> np.ndarray:
    """Sub-sample fiducial per coarse peak on the spline at 1 ms resolution."""
    lo, hi = t[pk - 1], t[pk + 1]
    grid = np.minimum(lo[:, None] + np.arange(0, 121)[None, :], hi[:, None])
    ts = grid[np.arange(len(pk)), np.argmax(cs(grid, 1), axis=1)]
    if kind == "maxslope":
        return ts
    if kind != "tangent":
        raise ValueError(f"unknown fiducial {kind!r}")
    back = np.maximum(ts[:, None] - np.arange(400, -1, -2)[None, :], t[0])  # preceding minimum, 400 ms
    tmin = back[np.arange(len(ts)), np.argmin(cs(back), axis=1)]
    slope = cs(ts, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = ts - (cs(ts) - cs(tmin)) / slope
    return np.where(np.isfinite(tf) & (slope > 0) & (tf >= tmin) & (tf <= ts), tf, ts)


def _znorm(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


TEMPLATE_STEP = 4  # the running template is recomputed every this many beats


def _running_template(snip: np.ndarray, n_tmpl: int) -> np.ndarray:
    """Per beat, the z-scored median of the z-scored snippets of its n_tmpl neighbours (computed
    at every TEMPLATE_STEP-th beat and shared by the beats around it)."""
    z = _znorm(snip)
    n = len(z)
    w = min(n_tmpl, n)
    centres = np.arange(TEMPLATE_STEP // 2, n + TEMPLATE_STEP // 2, TEMPLATE_STEP).clip(0, n - 1)
    start = np.clip(centres - w // 2, 0, n - w)
    tm = np.median(z[start[:, None] + np.arange(w)[None, :]], axis=1)
    return _znorm(tm[np.minimum(np.arange(n) // TEMPLATE_STEP, len(tm) - 1)])


def template_corr(cs: interpolate.CubicSpline, t: np.ndarray, tf: np.ndarray, win_ms: tuple[float, float], n_tmpl: int) -> np.ndarray:
    """Pearson correlation of each beat's snippet with the running median template (no shift)."""
    tau = np.arange(win_ms[0], win_ms[1] + 1e-9, 2.0)
    snip = _znorm(cs(np.clip(tf[:, None] + tau[None, :], t[0], t[-1])))
    return (snip * _running_template(snip, n_tmpl)).sum(1)


def template_align(
    cs: interpolate.CubicSpline,
    t: np.ndarray,
    tf: np.ndarray,
    win_ms: tuple[float, float],
    qual_win_ms: tuple[float, float],
    n_tmpl: int,
    max_shift_ms: float,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align each beat to the running median template of its neighbours (1 ms grid).
    Returns (aligned times, quality correlation; -1 where the timing peak hit the search edge)."""
    tau = np.arange(win_ms[0], win_ms[1] + 1e-9, 1.0)
    L = len(tau)
    M = int(round(max_shift_ms))
    ext = np.arange(win_ms[0] - M, win_ms[1] + M + 1e-9, 1.0)
    lo, hi = t[0], t[-1]
    tf = tf.copy()
    inner = np.ones(len(tf), bool)
    rows = np.arange(len(tf))
    for _ in range(n_iter):
        tmpl = _running_template(cs(np.clip(tf[:, None] + tau[None, :], lo, hi)), n_tmpl)
        s = cs(np.clip(tf[:, None] + ext[None, :], lo, hi))
        num = signal.fftconvolve(s, tmpl[:, ::-1], mode="valid", axes=1)  # (n, 2M + 1)
        c1 = np.cumsum(np.c_[np.zeros(len(s)), s], axis=1)
        c2 = np.cumsum(np.c_[np.zeros(len(s)), s * s], axis=1)
        sm, sq = c1[:, L:] - c1[:, :-L], c2[:, L:] - c2[:, :-L]
        r = num / np.sqrt(np.maximum(sq - sm * sm / L, 1e-24))  # tmpl is zero-mean, unit-norm
        k = np.argmax(r, axis=1)
        inner = (k > 0) & (k < 2 * M)
        kk = np.clip(k, 1, 2 * M - 1)
        y0, y1, y2 = r[rows, kk - 1], r[rows, kk], r[rows, kk + 1]
        den = y0 - 2 * y1 + y2
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(inner & (den < 0), 0.5 * (y0 - y2) / den, 0.0)
        tf = tf + (k - M) + np.clip(delta, -0.5, 0.5)
    corr = template_corr(cs, t, tf, qual_win_ms, n_tmpl)
    return tf, np.where(inner, corr, -1.0)


def motion_level(acc_t_ms: np.ndarray, acc_xyz: np.ndarray, win_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    """(t, RMS over win_s of the 0.5-10 Hz band-passed accelerometer magnitude), m/s^2."""
    at = np.asarray(acc_t_ms, dtype=np.float64)
    if len(at) < 4 * FS:
        return None
    mag = np.linalg.norm(np.asarray(acc_xyz, dtype=np.float64), axis=1)
    sos = signal.butter(2, (0.5, 10.0), "bandpass", fs=FS, output="sos")
    out = np.zeros(len(mag))
    cuts = np.r_[0, np.where(np.diff(at) > GAP_MS)[0] + 1, len(at)]
    k = max(1, int(round(win_s * FS)))
    for a, b in itertools.pairwise(cuts):
        if b - a < 30:
            out[a:b] = np.inf
            continue
        f = signal.sosfiltfilt(sos, mag[a:b] - mag[a:b].mean())
        out[a:b] = np.sqrt(np.convolve(f * f, np.ones(k) / k, mode="same"))
    return at, out


def artifact_correct(
    bt: np.ndarray, q: np.ndarray, c: np.ndarray, med_beats: int = 11, short: float = 0.7, long: float = 1.4, ddev: float = 4.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean a beat series (times bt, quality q 0/1, correlation c); returns new sorted arrays.

    Works on the good beats (q > 0) and their local median interval (`med_beats`):
    - extra beat: an interval < short x median; the end beat whose removal makes the merged
      interval closest to the median is removed if that merge is within 30 % of it, otherwise
      both ends are rejected (q = 0);
    - long interval: > long x median or > 2 s between consecutive good beats gets a quality-0
      placeholder in the middle (corr NaN), so neither it nor a merge across it is used;
    - misplaced beat: r_j = t_j - (t_j-1 + t_j+1) / 2 (half the jump between its two intervals,
      both real); a beat with |r_j| > ddev x the local robust SD of r (91 beats) that is the
      largest of its neighbours is rejected."""
    bt, q, c = bt.copy(), q.astype(float).copy(), c.copy()
    for _ in range(2):  # extra beats
        good = np.where(q > 0)[0]
        if len(good) < med_beats + 2:
            break
        ibi = np.diff(bt[good])
        med = ndimage.median_filter(ibi, size=med_beats, mode="nearest")
        remove, reject = [], []
        j = 0
        while j < len(ibi):
            if ibi[j] < short * med[j]:
                m_prev = ibi[j - 1] + ibi[j] if j > 0 else np.inf  # remove good[j]
                m_next = ibi[j] + ibi[j + 1] if j + 1 < len(ibi) else np.inf  # remove good[j + 1]
                e_prev, e_next = abs(m_prev - med[j]), abs(m_next - med[j])
                if min(e_prev, e_next) <= 0.3 * med[j]:
                    remove.append(good[j] if e_prev < e_next else good[j + 1])
                else:
                    reject += [good[j], good[j + 1]]
                j += 2
            else:
                j += 1
        if not remove and not reject:
            break
        q[np.array(reject, dtype=int)] = 0.0
        keep = np.ones(len(bt), bool)
        keep[np.array(remove, dtype=int)] = False
        bt, q, c = bt[keep], q[keep], c[keep]
    good = np.where(q > 0)[0]
    if len(good) < med_beats + 2:
        return bt, q, c
    # long intervals: placeholder (before the misplaced-beat test, which would flag their ends)
    tg = bt[good]
    ibi = np.diff(tg)
    med = ndimage.median_filter(ibi, size=med_beats, mode="nearest")
    gap = ((ibi > long * med) | (ibi > 2000.0)) & (np.diff(good) == 1)
    ph = (tg[:-1] + tg[1:])[gap] / 2.0
    if len(ph):
        bt = np.r_[bt, ph]
        q = np.r_[q, np.zeros(len(ph))]
        c = np.r_[c, np.full(len(ph), np.nan)]
        o = np.argsort(bt, kind="stable")
        bt, q, c = bt[o], q[o], c[o]
    for _ in range(2):  # misplaced beats
        good = np.where(q > 0)[0]
        tg = bt[good]
        both = (np.diff(good[:-1]) == 1) & (np.diff(good[1:]) == 1)  # interior beats with two real intervals
        r = np.full(len(tg), np.nan)
        r[1:-1] = np.where(both, tg[1:-1] - (tg[:-2] + tg[2:]) / 2.0, np.nan)
        ar = np.abs(r)
        if np.isfinite(ar).sum() < 10:
            break
        filled = np.where(np.isfinite(ar), ar, np.nanmedian(ar))
        sd = 1.4826 * ndimage.median_filter(filled, size=min(91, len(filled)), mode="nearest") + 1.0
        nb = np.fmax(np.r_[0.0, filled[:-1]], np.r_[filled[1:], 0.0])
        bad = np.isfinite(ar) & (ar > ddev * sd) & (filled >= nb)
        if not bad.any():
            break
        q[good[bad]] = 0.0
    return bt, q, c


def fit_drops(
    bt: np.ndarray,
    good: np.ndarray,
    si: np.ndarray,
    drop_ms: float = 40.0,
    n_range: tuple[float, float] = (150.0, 520.0),
    jitter: float = 15.0,
    g_min: float = 0.0,
    cap_ms: float = 60.0,
    skip_pen: float = 10.0,
    bin_pen: float = 5.0,
    pick_cost: float = 10.0,
) -> tuple[np.ndarray, int]:
    """Find the watch's silently lost PPG samples; return the per-beat shift that undoes them.

    On GalaxyPPG the watch delivers fewer PPG samples than its 25 Hz sensor takes: about one in
    every 150-500 is lost, while the stamps stay evenly spaced (~40.1-40.2 ms on average). On the
    stamp clock the beat times then carry a sawtooth, a slow ramp and a one-period jump at each
    lost sample, so the interval spanning a loss comes out one period (`drop_ms`) short (seen
    against the ECG on the development participants: ~7 % of resting intervals ~40 ms short, the
    losses recurring every ~6-15 s). Losses recur at a slowly drifting period in sample index.

    `bt` are beat times, `good` the usable ones, `si` their positions in delivered-sample index.
    Evidence per interval k between consecutive good beats: g(k) = -(delta_k + drop_ms / 2), where
    delta_k = interval minus the mean of its two neighbours (one-sided at breaks), clipped to
    [-2 drop_ms, cap_ms]; ~ +drop_ms / 2 for a loss, ~ -drop_ms / 2 otherwise. A Viterbi pass picks
    the chain of losses one period apart (+-`jitter` samples; period bins 2 % apart, moving at most
    one bin per loss at `bin_pen`; one unseen loss costs `skip_pen`; each loss costs `pick_cost`; a
    new chain may start anywhere) that maximises the summed evidence. Losses on the chain whose
    evidence is > g_min are undone: every later beat moves by drop_ms, minus the matching ramp so
    the long-run clock stays the stamps'. Returns (shift_ms per beat, number of losses undone)."""
    shift = np.zeros(len(bt))
    gi = np.where(good)[0]
    if len(gi) < 20:
        return shift, 0
    tg, sg = bt[gi], si[gi]
    ibi = np.diff(tg)
    valid = np.diff(gi) == 1  # no rejected beat between: a real interval
    prev_ok, next_ok = np.r_[False, valid[:-1]], np.r_[valid[1:], False]
    ip, inx = np.r_[np.nan, ibi[:-1]], np.r_[ibi[1:], np.nan]
    nb = np.where(prev_ok & next_ok, (ip + inx) / 2.0, np.where(prev_ok, ip, np.where(next_ok, inx, np.nan)))
    with np.errstate(invalid="ignore"):
        delta = np.clip(ibi - nb, -2.0 * drop_ms, cap_ms)
    g = np.where(valid & np.isfinite(nb), -(delta + drop_ms / 2.0), -np.inf)
    c = (sg[:-1] + sg[1:]) / 2.0
    path = _drop_chain(c, g - pick_cost, n_range, jitter, skip_pen, bin_pen)
    path = path[g[path] > g_min]
    if not len(path):
        return shift, 0
    p = np.sort(sg[path + 1] - 0.5)  # the loss sits inside interval k: beats from gi[k + 1] on move
    count = np.searchsorted(p, si, side="right").astype(float)
    # count - 0.5 at the midpoint between losses; held at -0.5 / +0.5 outside the chain, so the
    # first and last losses get their full step and beats far from any loss keep their stamps
    ramp = np.interp(si, p, np.arange(len(p)) + 0.5)
    return drop_ms * (count - ramp), int(len(p))


def _drop_chain(
    c: np.ndarray, g: np.ndarray, n_range: tuple[float, float], jitter: float, skip_pen: float, bin_pen: float, restart_pen: float = 30.0
) -> np.ndarray:
    """Viterbi over candidate intervals (centres c on the sample axis, sorted; gains g, -inf =
    unusable). State (k, n): the latest loss is in interval k and the period is bin n. A loss
    follows the previous one by m periods (m = 2 when one loss went unseen, at skip_pen), +-jitter;
    the period may move one bin per loss (bin_pen). A chain may restart after any earlier state at
    least one minimum period back (restart_pen), so one pass covers a whole segment with breaks.
    Returns the chosen interval indices."""
    ratio = 1.02
    Ns = n_range[0] * ratio ** np.arange(int(np.log(n_range[1] / n_range[0]) / np.log(ratio)) + 1)
    K, B = len(c), len(Ns)
    if K == 0:
        return np.zeros(0, dtype=int)
    cols = np.arange(B)
    V = np.full((K, B), -np.inf)
    back = np.full((K, B, 2), -1, dtype=np.int64)
    Vf = np.full((K, B), -np.inf)  # best over neighbouring period bins, with the change penalty
    arg_n = np.zeros((K, B), dtype=np.int64)
    row_best = np.full(K, -np.inf)
    row_arg = np.zeros(K, dtype=np.int64)
    pre_best = np.full(K, -np.inf)  # prefix max of row_best
    pre_arg = np.full(K, -1, dtype=np.int64)
    # predecessor ranges [lo, hi) per (k, period bin) for one and two periods back
    bounds = []
    for m in (1, 2):
        half = jitter * (1.0 + 0.5 * (m - 1))
        lo = np.searchsorted(c, c[:, None] - m * Ns[None, :] - half, side="left")
        hi = np.minimum(np.searchsorted(c, c[:, None] - m * Ns[None, :] + half, side="right"), np.arange(K)[:, None])
        bounds.append((m, lo, hi, int(max(1, (hi - lo).max(initial=0)))))
    restart_row = np.searchsorted(c, c - Ns[0], side="right") - 1
    for k in range(K):
        if np.isfinite(g[k]):
            best = np.zeros(B)  # start the first chain here
            bk = np.full(B, -1)
            bn = np.full(B, -1)
            r = int(restart_row[k])
            if r >= 0 and pre_arg[r] >= 0 and pre_best[r] - restart_pen > 0:
                best[:] = pre_best[r] - restart_pen
                bk[:] = pre_arg[r]
                bn[:] = row_arg[pre_arg[r]]
            for m, lo_m, hi_m, span in bounds:
                lo, hi = lo_m[k], hi_m[k]
                for o in range(span):
                    kp = lo + o
                    ok = kp < hi
                    if not ok.any():
                        break
                    kp[~ok] = 0
                    cand = Vf[kp, cols]
                    cand[~ok] = -np.inf
                    if m == 2:
                        cand -= skip_pen
                    upd = cand > best
                    best[upd] = cand[upd]
                    bk[upd] = kp[upd]
                    bn[upd] = arg_n[kp[upd], cols[upd]]
            v = g[k] + best
            V[k] = v
            back[k, :, 0], back[k, :, 1] = bk, bn
            # best over this bin and its two neighbours, moving costs bin_pen
            vf, an = v.copy(), cols.copy()
            left, right = v[:-1] - bin_pen, v[1:] - bin_pen  # from bin n - 1 into n, from n + 1 into n
            u = left > vf[1:]
            vf[1:][u], an[1:][u] = left[u], cols[:-1][u]
            u = right > vf[:-1]
            vf[:-1][u], an[:-1][u] = right[u], cols[1:][u]
            Vf[k], arg_n[k] = vf, an
            row_arg[k] = int(np.argmax(v))
            row_best[k] = v[row_arg[k]]
        if k and pre_best[k - 1] >= row_best[k]:
            pre_best[k], pre_arg[k] = pre_best[k - 1], pre_arg[k - 1]
        elif np.isfinite(row_best[k]):
            pre_best[k], pre_arg[k] = row_best[k], k
    if pre_arg[-1] < 0:
        return np.zeros(0, dtype=int)
    k = int(pre_arg[-1])
    n = int(row_arg[k])
    path = []
    while k >= 0:
        path.append(k)
        k, n = (int(x) for x in back[k, n])
    return np.array(path[::-1], dtype=int)


def accept_windows(bt: np.ndarray, good: np.ndarray, ppg_t: np.ndarray, win_s: float, min_good: float) -> list[tuple[float, float]]:
    """Stretches where, in a sliding `win_s` window (step win_s / 6), good beats >= min_good x the
    beats expected from the median good interval. Returns merged (start_ms, end_ms) pairs."""
    tg = bt[good]
    if len(tg) < 3:
        return []
    rr = float(np.median(np.diff(tg)))
    W = win_s * 1000.0
    step = W / 6.0
    starts = np.arange(ppg_t[0], ppg_t[-1] - W + step, step)
    if not len(starts):
        return []
    n_good = np.searchsorted(tg, starts + W) - np.searchsorted(tg, starts)
    out: list[tuple[float, float]] = []
    for s, ok in zip(starts, n_good >= min_good * (W / rr)):
        if ok:
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], s + W)
            else:
                out.append((s, s + W))
    return out
