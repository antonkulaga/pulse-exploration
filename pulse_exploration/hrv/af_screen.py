"""Irregular-rhythm (atrial fibrillation) screen on beat intervals.

Works on any beat source: ECG R-peaks, PPG beats from the 25 Hz watch stream, or a cleaned interval
list. Intervals are cut into 64-beat segments ("tachograms") of consecutive usable beats. A segment
is irregular when both of these exceed the thresholds Lee et al. derived on the MIT-BIH AF and
normal-rhythm databases for a 30 fps smartphone camera (IEEE TBME 60(1):203, 2013):
- RMSSD / mean interval >= 0.13, here with the 8 largest successive differences left out
  (`TRIM_DIFFS`), so that a few missed, extra or mistimed beats in sinus rhythm do not look like
  AF. On GalaxyPPG's 25 Hz beats this cut false irregular segments from 9.5 % to 0 % (696
  segments, nobody in AF), while synthetic AF stayed 100 % flagged (median 0.21). The threshold
  was derived without trimming, so it should be re-checked on real AF recordings.
- Shannon entropy of the intervals >= 0.76: the 8 longest and 8 shortest intervals are dropped,
  the rest go into 16 equal-width bins, and the entropy is divided by log(16) (Dash et al., Ann
  Biomed Eng 37:1701, 2009)

An alert needs persistence, as in the cleared watch features: at least `alert_k` irregular segments
among `alert_n` consecutive ones. Sinus arrhythmia, ectopic beats and PPG timing errors are the
known causes of false alerts; the segment features are returned so they can be inspected.

Research prototype; a positive screen calls for a 30 s ECG, not a diagnosis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SEGMENT = 64
TRIM = 8
TRIM_DIFFS = 8
BINS = 16
TH_NRMSSD = 0.13
TH_SHANNON = 0.76
# Beat settings for the screen: foot_template's interval-based artifact correction treats an
# irregular rhythm as artefacts and, on synthetic AF pulse waves, left no 64-beat segment at all.
# The screen keeps the beat-shape and motion checks and the lost-sample fix.
AF_SETTINGS = {"art": False}


def nrmssd(ibi: np.ndarray, trim: int = TRIM_DIFFS) -> float:
    """RMSSD / mean interval, leaving out the `trim` largest absolute successive differences."""
    ibi = np.asarray(ibi, dtype=np.float64)
    d = np.sort(np.abs(np.diff(ibi)))
    if trim and len(d) > trim:
        d = d[: len(d) - trim]
    return float(np.sqrt(np.mean(d ** 2)) / np.mean(ibi))


def shannon_entropy(ibi: np.ndarray, trim: int = TRIM, bins: int = BINS) -> float:
    x = np.sort(np.asarray(ibi, dtype=np.float64))
    if len(x) > 2 * trim + 1:
        x = x[trim:len(x) - trim]
    if np.ptp(x) == 0:
        return 0.0
    counts, _ = np.histogram(x, bins=bins)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(bins))


@dataclass(frozen=True)
class Segment:
    start_ms: float  # time of the first beat
    end_ms: float  # time of the last beat
    nrmssd: float
    shannon: float
    mean_ibi_ms: float
    irregular: bool


def segments_from_beats(beats_ms: np.ndarray, good: np.ndarray | None = None, n: int = SEGMENT,
                        th_nrmssd: float = TH_NRMSSD, th_shannon: float = TH_SHANNON,
                        max_ibi_ms: float = 2000.0) -> list[Segment]:
    """Non-overlapping segments of `n` consecutive intervals between usable beats. A rejected beat,
    or an interval over `max_ibi_ms` (a gap), ends the current run."""
    b = np.asarray(beats_ms, dtype=np.float64)
    ok = np.ones(len(b), bool) if good is None else np.asarray(good, bool)
    ibi = np.diff(b)
    usable = ok[1:] & ok[:-1] & (ibi > 0) & (ibi <= max_ibi_ms)
    return _segments(b[1:], ibi, usable, n, th_nrmssd, th_shannon)


def segments_from_intervals(end_ms: np.ndarray, ibi_ms: np.ndarray, ok: np.ndarray, adjacent: np.ndarray,
                            n: int = SEGMENT, th_nrmssd: float = TH_NRMSSD,
                            th_shannon: float = TH_SHANNON) -> list[Segment]:
    """Same, for an interval list (e.g. Samsung's IBIs): `adjacent[j]` says intervals j and j+1
    are consecutive beats, so a non-adjacent pair ends the current run."""
    e = np.asarray(end_ms, dtype=np.float64)
    v = np.asarray(ibi_ms, dtype=np.float64)
    usable = np.asarray(ok, bool).copy()
    # break runs at non-adjacent pairs: mark the second interval of such a pair as a run start
    brk = np.r_[False, ~np.asarray(adjacent, bool)[: len(v) - 1]]
    return _segments(e, v, usable, n, th_nrmssd, th_shannon, brk)


def _segments(t_end: np.ndarray, ibi: np.ndarray, usable: np.ndarray, n: int, th_r: float, th_s: float,
              breaks: np.ndarray | None = None) -> list[Segment]:
    out: list[Segment] = []
    run: list[int] = []
    for j in range(len(ibi)):
        if not usable[j] or (breaks is not None and breaks[j]):
            run = [j] if usable[j] else []
            continue
        run.append(j)
        if len(run) == n:
            seg = ibi[run]
            r, s = nrmssd(seg), shannon_entropy(seg)
            out.append(Segment(float(t_end[run[0]] - ibi[run[0]]), float(t_end[run[-1]]), r, s,
                               float(seg.mean()), bool(r >= th_r and s >= th_s)))
            run = []
    return out


def alerts(segs: list[Segment], alert_k: int = 5, alert_n: int = 6, max_gap_ms: float = 600_000.0) -> list[tuple[float, float]]:
    """Alert episodes: `alert_k` irregular segments among `alert_n` consecutive ones, where
    consecutive segments are at most `max_gap_ms` apart. Returns (start_ms, end_ms) per alert;
    overlapping windows are merged."""
    out: list[tuple[float, float]] = []
    for i in range(len(segs) - alert_n + 1):
        win = segs[i:i + alert_n]
        if any(win[k + 1].start_ms - win[k].end_ms > max_gap_ms for k in range(alert_n - 1)):
            continue
        if sum(s.irregular for s in win) >= alert_k:
            a, b = win[0].start_ms, win[-1].end_ms
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
    return out
