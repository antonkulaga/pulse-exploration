"""25 Hz candidate: accelerometer motion cancellation, DP beat tracking with sub-sample upstroke
timing, artifact handling (incl. the stream's hidden sample slips) and a heavy-motion gate.

Pipeline, per contiguous PPG segment (split at gaps > 250 ms; dropped samples inside a segment are
filled on an integer sample grid, beat times map back through the real timestamps):

1. Motion cancellation. -PPG (the green ADC falls as blood volume rises) and the three accelerometer
   axes are band-passed 0.5-5 Hz. Block least-squares (Wiener) cancellation: in 16 s, 50 %-overlap
   Hann blocks the PPG is ridge-regressed on 16 taps (8 past, 8 future) of each axis and the fit is
   subtracted. It is crossfaded in only where the 4 s acc RMS exceeds 0.1-0.3 m/s^2: ungated, any
   accelerometer regression (also NLMS) damages the resting PPG (dev RMSSD error 5.3 -> 16 ms).
   NLMS (`denoise="nlms"`) is kept as an option; it never beat no cancellation on the dev set.
2. Robust beat detection. Candidates are local maxima of the slope (upstroke) above 0.15 of a
   slope envelope (2 s max, 8 s median). The expected period comes from an 8 s autocorrelation
   track (half-lag preferred when nearly as strong). Dynamic programming picks the candidate
   sequence maximising sum(score) - 4 log(interval / k RR)^2 - 0.6 (k - 1) per link (k - 1 missed
   beats), with a jump (cost 3) out of untrackable stretches. This rejects dicrotic-wave slopes.
3. Sub-sample timing. x10 band-limited upsampling (polyphase Kaiser FIR), the slope maximum on the
   fine grid, a parabola through it. Max slope beat the tangent foot, the 30 %/50 % amplitude point,
   the systolic peak and template cross-correlation at seated rest (IBI error RMS 13.9 ms vs 14.4-17.6).
4. Artifact handling. Beat-level (the beat and both its intervals unusable, quality 0): slope score
   < 0.35; pulse-shape correlation with the running median of 21 neighbours < 0.3, or < 0.7 when the
   beat also sits > 25 ms off the midpoint of its neighbours; beat-time residual > max(3 robust SD,
   30 ms); band-passed acc RMS within +-0.5 s > 0.6 m/s^2. Interval-level (a quality-0 placeholder
   beat inside the interval voids only that interval): links that skipped beats, recording gaps,
   intervals > 20 % off the local median, and hidden sample slips (below).
5. Gate. Output is vouched for (`accept_windows`) only where the sustained motion level (10 s
   running median of the 1 s acc RMS) is <= 1.5 m/s^2: typing reaches <= 0.95, walking >= 2.15,
   jogging/running 14-19 on the dev set.

Hidden sample slips (data finding, affects every 25 Hz candidate): the GalaxyPPG 25 Hz stream drops
one content sample every ~10-12 s (5-15 s, not periodic) while its timestamps stay regular. Beat
times read off the timestamps therefore carry a 0-40 ms sawtooth against the ECG (ramp ~3-4 ms/s,
then a one-sample drop); each drop shortens one IBI by ~40 ms. The slips are not visible in the
timestamps, the batch boundaries or (reliably) the waveform; they show as an isolated short interval.
`slip="void"` voids intervals > 30 ms shorter than the mean of their two neighbours (at most one per
4 s). On the dev set this cuts the 60 s RMSSD median |error| 5.9 -> 3.5 ms and the bias +5.3 ->
+1.0 ms; a control voiding long isolated intervals instead (which slips cannot cause) gets 4.9 ms,
so about 1 ms of the gain is generic trimming. An ECG oracle that corrects every slip gets 3.4 ms.

Tuned on P02-P06 only. Output also carries `flags` (bit mask per beat, F_* below) for diagnostics.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, median_filter

FS = 25.0
T_MS = 1000.0 / FS

# per-beat flag bits (placeholders carry F_GAP, F_DEV or F_SLIP)
F_AMP, F_CORR, F_RESID, F_DEV, F_GAP, F_SLIP, F_ACC = 1, 2, 4, 8, 16, 32, 64


@dataclass(frozen=True)
class Params:
    gap_ms: float = 250.0  # split segments where samples are missing for longer
    min_seg_s: float = 10.0
    band_hz: tuple[float, float] = (0.5, 5.0)
    order: int = 2
    # 1. motion cancellation
    denoise: str = "block"  # "block" | "nlms" | "none"
    denoise_gate: float = 0.2  # m/s^2: crossfade 0 -> 1 as the 4 s acc RMS goes 0.5 -> 1.5 x gate
    taps: int = 16  # per accelerometer axis
    lead: int = 8  # of which future samples (non-causal FIR)
    block_s: float = 16.0
    ridge: float = 0.1  # relative to the mean regressor power
    mu: float = 0.02  # NLMS only
    eps_acc: float = 0.2  # NLMS only: m/s^2, references weaker than this barely adapt
    # 2. beat tracking
    acf_win_s: float = 8.0
    acf_hop_s: float = 2.0
    cand_min: float = 0.15  # candidate slope peak / envelope
    lam: float = 4.0  # weight of log(interval / (k RR))^2
    miss_pen: float = 0.6  # per beat missed inside a link
    jump_pen: float = 3.0  # restart after an untrackable stretch
    rr_lo: float = 0.55  # link lengths considered, x expected RR
    rr_hi: float = 3.5
    # 3. timing
    up: int = 10
    # 4. artifacts
    artifact: bool = True
    q_amp: float = 0.35
    tmpl_beats: int = 21
    q_corr: float = 0.3  # shape alone flags a beat below this
    q_corr_soft: float = 0.7  # ... and below this when the beat residual also exceeds resid_soft_ms
    resid_soft_ms: float = 25.0
    resid_k: float = 3.0
    resid_floor_ms: float = 30.0
    dev_max: float = 0.2
    acc_beat_max: float = 0.6  # m/s^2, band-passed acc RMS within +-acc_beat_s of the beat
    acc_beat_s: float = 0.5
    slip: str = "void"  # "void" | "none"
    slip_thr_ms: float = 30.0
    slip_min_gap_s: float = 4.0
    # 5. gate
    heavy_acc: float = 1.5  # m/s^2, sustained motion above which nothing is vouched for
    motion_win_s: float = 10.0


class MotionDenoiseEstimator:
    """`bench.Estimator`: estimate(ppg_t_ms, ppg, acc_t_ms, acc_xyz) -> {beats_ms, quality,
    accept_windows, flags}. Keyword arguments override `Params` fields."""

    name = "motion_denoise"

    def __init__(self, name: str | None = None, **overrides):
        self.p = replace(Params(), **overrides)
        if name:
            self.name = name

    def estimate(self, ppg_t_ms: np.ndarray, ppg: np.ndarray, acc_t_ms: np.ndarray, acc_xyz: np.ndarray) -> dict:
        p = self.p
        t_all = np.asarray(ppg_t_ms, dtype=np.float64)
        x_all = np.asarray(ppg, dtype=np.float64)
        fin = np.isfinite(t_all) & np.isfinite(x_all)
        t_all, x_all = t_all[fin], x_all[fin]
        acc_t = np.asarray(acc_t_ms, dtype=np.float64)
        acc_xyz = np.asarray(acc_xyz, dtype=np.float64).reshape(-1, 3)
        if len(acc_t) and len(t_all):
            ok = np.isfinite(acc_t) & np.isfinite(acc_xyz).all(1)
            acc = np.column_stack([np.interp(t_all, acc_t[ok], acc_xyz[ok, j]) for j in range(3)])
        else:
            acc = np.zeros((len(t_all), 3))
        beats, qual, flags, accept = [], [], [], []
        for a, b in _segments(t_all, p.gap_ms, p.min_seg_s):
            tg, X = _regularize(t_all[a:b], np.column_stack([x_all[a:b], acc[a:b]]))
            s = self._segment(tg, X)
            if p.artifact and len(s["t"]) > 5:
                t, q, f = self._quality(s)
            else:
                t, q, f = s["t"], np.ones(len(s["t"])), np.zeros(len(s["t"]), int)
            if beats:  # a recording gap is not an interval: void it with a placeholder
                prev = next((bb[-1] for bb in reversed(beats) if len(bb)), tg[0])
                beats.append(np.array([0.5 * (prev + (t[0] if len(t) else tg[0]))]))
                qual.append(np.zeros(1))
                flags.append(np.array([F_GAP]))
            beats.append(t)
            qual.append(q)
            flags.append(f)
            ok = s["motion"] <= p.heavy_acc
            cuts = np.r_[0, np.where(np.diff(ok.astype(int)) != 0)[0] + 1, len(ok)]
            accept += [(float(tg[c0]), float(tg[c1 - 1])) for c0, c1 in itertools.pairwise(cuts) if ok[c0]]
        if not beats:
            return {"beats_ms": np.zeros(0), "quality": np.zeros(0), "accept_windows": [], "flags": np.zeros(0, int)}
        return {
            "beats_ms": np.concatenate(beats),
            "quality": np.concatenate(qual),
            "accept_windows": accept,
            "flags": np.concatenate(flags),
        }

    # ------------------------------------------------------------------------------------------
    def _segment(self, tg: np.ndarray, X: np.ndarray) -> dict:
        """Steps 1-3 on one regularised segment: cleaned signal, beats, scores, motion level."""
        p = self.p
        sos = signal.butter(p.order, p.band_hz, "bandpass", fs=FS, output="sos")
        x = signal.sosfiltfilt(sos, -X[:, 0])  # green ADC falls with blood volume: invert
        acc = signal.sosfiltfilt(sos, X[:, 1:4], axis=0)
        acc_mag = np.sqrt((acc**2).sum(1))
        motion = median_filter(_running_rms(acc_mag, int(FS)), size=max(1, int(p.motion_win_s * FS)), mode="nearest")
        if p.denoise == "block":
            est = block_cancel(x, acc, p.taps, p.lead, int(p.block_s * FS), p.ridge)
        elif p.denoise == "nlms":
            scale = np.std(x) + 1e-12
            est = nlms(x / scale, acc, p.taps, p.mu, p.taps * 3 * p.eps_acc**2, p.lead) * scale
        else:
            est = np.zeros_like(x)
        if p.denoise_gate > 0:
            est = est * np.clip(_running_rms(acc_mag, int(4 * FS)) / p.denoise_gate - 0.5, 0.0, 1.0)
        y = x - est
        d = np.gradient(y)
        env = median_filter(maximum_filter1d(d, int(2 * FS)), int(8 * FS), mode="nearest")
        env = np.maximum(env, 1e-9 * (np.abs(d).max() + 1e-12))
        i = np.where((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]))[0] + 1
        sc = d[i] / env[i]
        keep = sc >= p.cand_min
        ci, cs = i[keep], np.minimum(sc[keep], 1.5)
        cen, per = hr_track(d, p.acf_win_s, p.acf_hop_s)
        sel, ks = track_beats(ci.astype(np.float64), cs, np.interp(ci, cen, per), p)
        frac = refine_upstroke(y, ci[sel], p.up)
        return {
            "t": np.interp(frac, np.arange(len(tg)), tg),
            "frac": frac,
            "score": cs[sel],
            "k": ks,
            "rr_ms": np.interp(frac, cen, per) * T_MS,
            "acc_mag": acc_mag,
            "y": y,
            "motion": motion,
        }

    def _quality(self, s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step 4: per-beat quality, plus quality-0 placeholder beats that void single intervals."""
        p = self.p
        t, k = s["t"], s["k"]
        n = len(t)
        flags = np.zeros(n, int)
        flags[s["score"] < p.q_amp] |= F_AMP
        corr = _template_corr(s["y"], s["frac"], p.tmpl_beats)
        flags[corr < p.q_corr] |= F_CORR
        am = s["acc_mag"]
        acc_rms = _running_rms(am, 2 * int(p.acc_beat_s * FS) + 1)[np.clip(np.rint(s["frac"]).astype(int), 0, len(am) - 1)]
        flags[acc_rms > p.acc_beat_max] |= F_ACC
        ibi = np.diff(t)
        real = k[1:] == 1  # interval j (beat j -> j + 1) links consecutive beats
        med = _nan_medfilt(np.where(real, ibi, np.nan), 15)
        med = np.where(np.isfinite(med), med, s["rr_ms"][1:])
        dev = np.abs(ibi / med - 1)
        # beat residual: offset from the midpoint of its neighbours (= half the successive difference)
        e = np.zeros(n)
        e[1:-1] = np.where(real[:-1] & real[1:], (ibi[:-1] - ibi[1:]) / 2, np.nan)
        ae = np.nan_to_num(np.abs(e), nan=0.0)
        scale = np.nan_to_num(1.4826 * _nan_medfilt(np.abs(e[1:-1]), 31), nan=0.0)
        flags[1:-1][ae[1:-1] > np.maximum(p.resid_k * scale, p.resid_floor_ms)] |= F_RESID
        flags[(corr < p.q_corr_soft) & (ae > p.resid_soft_ms)] |= F_CORR
        q = np.where(flags > 0, 0.0, 1.0)
        ins_t: list[float] = []
        ins_f: list[int] = []
        for j in np.where(~real)[0]:  # skipped beats or a jump: one placeholder per missing beat
            m = max(1, int(round(ibi[j] / s["rr_ms"][j + 1])) - 1) if k[j + 1] >= 2 else 1
            ins_t.extend(t[j] + ibi[j] * np.arange(1, m + 1) / (m + 1))
            ins_f.extend([F_GAP] * m)
        for j in np.where(real & (dev > p.dev_max) & (q[:-1] > 0) & (q[1:] > 0))[0]:
            ins_t.append(t[j] + ibi[j] / 2)
            ins_f.append(F_DEV)
        if p.slip == "void":
            sj = slip_candidates(t, q, real & (dev <= p.dev_max), p.slip_thr_ms, p.slip_min_gap_s * 1000)
            ins_t.extend(t[sj] + ibi[sj] / 2)
            ins_f.extend([F_SLIP] * len(sj))
        if ins_t:
            t = np.r_[t, ins_t]
            q = np.r_[q, np.zeros(len(ins_t))]
            flags = np.r_[flags, ins_f]
            o = np.argsort(t, kind="stable")
            t, q, flags = t[o], q[o], flags[o]
        return t, q, flags


# --------------------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------------------


def _segments(t: np.ndarray, gap_ms: float, min_s: float) -> list[tuple[int, int]]:
    cuts = np.r_[0, np.where(np.diff(t) > gap_ms)[0] + 1, len(t)]
    return [(a, b) for a, b in itertools.pairwise(cuts) if b - a > 1 and (t[b - 1] - t[a]) >= min_s * 1000]


def _regularize(t: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put a segment on an integer sample index; samples missing from the timestamps (dt ~ 2-6 T)
    are filled linearly. Returns (timestamp of every grid sample, columns on the grid)."""
    steps = np.maximum(1, np.rint(np.diff(t) / T_MS)).astype(np.int64)
    k = np.r_[0, np.cumsum(steps)]
    g = np.arange(k[-1] + 1)
    return np.interp(g, k, t), np.column_stack([np.interp(g, k, cols[:, j]) for j in range(cols.shape[1])])


def _tap_matrix(refs: np.ndarray, taps: int, lead: int) -> np.ndarray:
    """(n, taps * m): row i holds refs[i - taps + 1 + lead .. i + lead] of every column (zero-padded)."""
    n, m = refs.shape
    pad = np.vstack([np.zeros((taps, m)), refs, np.zeros((taps, m))])
    win = np.lib.stride_tricks.sliding_window_view(pad, taps, axis=0)
    return np.ascontiguousarray(win[1 + lead : 1 + lead + n]).reshape(n, -1)


def block_cancel(x: np.ndarray, refs: np.ndarray, taps: int, lead: int, block: int, ridge: float) -> np.ndarray:
    """Block least-squares (Wiener) estimate of the reference-correlated part of x: ridge regression
    on the tapped references in 50 %-overlapping Hann-weighted blocks, overlap-added."""
    n = len(x)
    U = _tap_matrix(refs, taps, lead)
    est, wsum = np.zeros(n), np.zeros(n)
    block = max(block, 4 * U.shape[1])
    hop = block // 2
    starts = list(range(0, max(n - block, 0) + 1, hop)) or [0]
    if starts[-1] + block < n:
        starts.append(n - block)
    win = np.hanning(block + 2)[1:-1]
    eye = np.eye(U.shape[1])
    for s0 in starts:
        sl = slice(s0, min(s0 + block, n))
        A, b = U[sl], x[sl]
        G = A.T @ A
        lam = ridge * np.trace(G) / G.shape[0] + 1e-9 * len(b)
        w = np.linalg.solve(G + lam * eye, A.T @ b)
        ww = win[: len(b)]
        est[sl] += ww * (A @ w)
        wsum[sl] += ww
    return est / np.maximum(wsum, 1e-9)


def nlms(x: np.ndarray, refs: np.ndarray, taps: int, mu: float, eps: float, lead: int) -> np.ndarray:
    """Normalised LMS estimate of the reference-correlated part of x (same tap layout as
    `_tap_matrix`); eps regularises the normalisation so weak references barely adapt."""
    U = _tap_matrix(refs, taps, lead)
    nrm = eps + np.einsum("ij,ij->i", U, U)
    w = np.zeros(U.shape[1])
    est = np.empty(len(x))
    for i in range(len(x)):
        u = U[i]
        est[i] = w @ u
        w += (mu * (x[i] - est[i]) / nrm[i]) * u
    return est


def _running_rms(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(x)))
    return np.sqrt(np.convolve(x * x, np.ones(k) / k, mode="same")) + 1e-12


def _parabola(y: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Fractional index of the maximum through y[m - 1], y[m], y[m + 1]."""
    m = np.clip(m, 1, len(y) - 2)
    y0, y1, y2 = y[m - 1], y[m], y[m + 1]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den < 0, 0.5 * (y0 - y2) / den, 0.0)
    return m + np.clip(d, -0.5, 0.5)


def hr_track(d: np.ndarray, win_s: float, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Local beat period (samples) from the autocorrelation of the slope signal in sliding windows:
    (window centres, periods), median-smoothed over 5 windows. The half lag wins when it reaches
    0.6 of the strongest peak (guards against locking onto every second beat)."""
    W, H = min(int(win_s * FS), len(d)), int(hop_s * FS)
    lo, hi = int(0.3 * FS), int(2.0 * FS)
    nfft = 1 << int(np.ceil(np.log2(2 * max(W, 2))))
    cen, per = [], []
    for s0 in range(0, max(len(d) - W, 0) + 1, H):
        seg = d[s0 : s0 + W] - d[s0 : s0 + W].mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nfft)) ** 2)[: hi + 2]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        pk, _ = signal.find_peaks(ac[lo : hi + 1])
        if not len(pk):
            continue
        L = int(pk[np.argmax(ac[lo + pk])] + lo)
        near = [int(q + lo) for q in pk if abs(q + lo - L / 2) <= 2]
        if L / 2 >= lo and near and max(ac[v] for v in near) >= 0.6 * ac[L]:
            L = max(near, key=lambda v: ac[v])
        cen.append(s0 + W / 2)
        per.append(float(_parabola(ac, np.array([L]))[0]))
    if not per:
        return np.array([len(d) / 2]), np.array([0.8 * FS])
    return np.asarray(cen), median_filter(np.asarray(per), size=5, mode="nearest")


def track_beats(tc: np.ndarray, sc: np.ndarray, rr: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic-programming beat tracking over candidate upstrokes.

    tc: candidate positions (samples, sorted), sc: scores (~1 for a clean upstroke), rr: expected
    period at each candidate (samples). A link of r = interval / rr spanning k = round(r) beats
    costs lam log(r / k)^2 + miss_pen (k - 1); links longer than rr_hi rr jump from the best earlier
    path at jump_pen. Returns (chosen candidate indices, k of the link into each; 0 = start/jump)."""
    n = len(tc)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    F = np.empty(n)
    P = np.full(n, -1)
    K = np.zeros(n, int)
    best_f = np.empty(n)  # running max of F[0..i] and where it is
    best_i = np.empty(n, int)
    for i in range(n):
        R, ti = rr[i], tc[i]
        j0 = np.searchsorted(tc, ti - p.rr_hi * R, "left")
        j1 = np.searchsorted(tc, ti - p.rr_lo * R, "right")
        f, bj, bk = sc[i], -1, 0
        if j0 > 0 and best_f[j0 - 1] - p.jump_pen + sc[i] > f:
            f, bj, bk = best_f[j0 - 1] - p.jump_pen + sc[i], int(best_i[j0 - 1]), 0
        if j1 > j0:
            js = np.arange(j0, j1)
            r = (ti - tc[js]) / R
            kk = np.maximum(1, np.rint(r))
            v = F[js] - p.lam * np.log(r / kk) ** 2 - p.miss_pen * (kk - 1) + sc[i]
            m = int(np.argmax(v))
            if v[m] > f:
                f, bj, bk = float(v[m]), int(js[m]), int(kk[m])
        F[i], P[i], K[i] = f, bj, bk
        if i == 0 or f >= best_f[i - 1]:
            best_f[i], best_i[i] = f, i
        else:
            best_f[i], best_i[i] = best_f[i - 1], best_i[i - 1]
    path, ks = [], []
    i = int(best_i[n - 1])
    while i >= 0:
        path.append(i)
        ks.append(K[i])
        i = P[i]
    return np.array(path[::-1], dtype=int), np.array(ks[::-1], dtype=int)


def refine_upstroke(y: np.ndarray, pk: np.ndarray, up: int) -> np.ndarray:
    """Fractional sample index of the steepest upstroke near each coarse peak `pk` of the slope:
    band-limited x`up` upsampling, slope maximum within +-1 sample, parabola."""
    if not len(pk):
        return np.zeros(0)
    zu = np.gradient(signal.resample_poly(y, up, 1, window=("kaiser", 8.0)))
    c = np.asarray(pk) * up
    win = np.clip(c[:, None] + np.arange(-up, up + 1)[None, :], 0, len(zu) - 1)
    m = win[np.arange(len(c)), np.argmax(zu[win], axis=1)]
    return _parabola(zu, m) / up


def slip_candidates(t: np.ndarray, q: np.ndarray, ok_iv: np.ndarray, thr_ms: float, min_gap_ms: float) -> np.ndarray:
    """Intervals j that look like a hidden dropped sample: ibi_j shorter than the mean of its two
    neighbours by more than thr_ms, all three usable; strongest first, at most one per min_gap_ms."""
    ibi = np.diff(t)
    if len(ibi) < 3:
        return np.zeros(0, int)
    good = ok_iv & (q[:-1] > 0) & (q[1:] > 0)
    imp = np.zeros(len(ibi))
    imp[1:-1] = ibi[1:-1] - 0.5 * (ibi[:-2] + ibi[2:])
    m = np.zeros(len(ibi), bool)
    m[1:-1] = good[1:-1] & good[:-2] & good[2:]
    cand = np.where(m & (imp < -thr_ms))[0]
    chosen: list[int] = []
    for j in cand[np.argsort(imp[cand])]:
        if all(abs(t[j] - t[c]) >= min_gap_ms for c in chosen):
            chosen.append(int(j))
    return np.array(sorted(chosen), dtype=int)


def _nan_medfilt(x: np.ndarray, size: int) -> np.ndarray:
    """Running median ignoring NaNs (NaN where a window holds none)."""
    if len(x) == 0:
        return x
    h = size // 2
    w = np.lib.stride_tricks.sliding_window_view(np.r_[np.full(h, np.nan), x, np.full(h, np.nan)], 2 * h + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(w, axis=1)


def _template_corr(y: np.ndarray, frac: np.ndarray, n_beats: int, lo_s: float = -0.2, hi_s: float = 0.48) -> np.ndarray:
    """Correlation of each pulse (lo_s..hi_s around the upstroke) with the running median of its
    n_beats neighbours."""
    fi = np.rint(frac).astype(int)
    win = np.clip(fi[:, None] + np.arange(int(lo_s * FS), int(hi_s * FS) + 1)[None, :], 0, len(y) - 1)
    snip = y[win] - y[win].mean(1, keepdims=True)
    snip /= np.linalg.norm(snip, axis=1, keepdims=True) + 1e-12
    tmpl = median_filter(snip, size=(n_beats, 1), mode="nearest")
    tmpl /= np.linalg.norm(tmpl, axis=1, keepdims=True) + 1e-12
    return (snip * tmpl).sum(1)
