"""Naive reference candidate: band-pass + scipy find_peaks on the raw 25 Hz PPG, no interpolation.

Exists to prove the benchmark end to end and to set the floor every real candidate must beat:
beats sit on the 40 ms sample grid, nothing handles motion, nothing is declined.

Polarity: the Galaxy Watch green ADC falls as blood volume rises (the alignment finds the steepest
post-R-peak change is a fall in every participant). Maxima of the raw signal are therefore pulse
feet (blood-volume minima), which are sharper at 25 Hz than the rounded systolic peaks; the default
detects them. `invert=True` detects systolic peaks instead (worse on GalaxyPPG).
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import signal

FS = 25.0
GAP_MS = 200.0  # split the stream where samples are missing


class NaiveEstimator:
    name = "naive"

    def __init__(
        self, band_hz: tuple[float, float] = (0.5, 4.0), min_distance_s: float = 0.3, prominence_sd: float = 0.5, invert: bool = False
    ):
        self.invert = invert
        if invert:
            self.name = "naive_inverted"
        self.band_hz = band_hz
        self.min_distance_s = min_distance_s
        self.prominence_sd = prominence_sd

    def estimate(self, ppg_t_ms: np.ndarray, ppg: np.ndarray, acc_t_ms: np.ndarray, acc_xyz: np.ndarray) -> dict:
        sos = signal.butter(2, self.band_hz, "bandpass", fs=FS, output="sos")
        cuts = np.r_[0, np.where(np.diff(ppg_t_ms) > GAP_MS)[0] + 1, len(ppg_t_ms)]
        beats = []
        for a, b in itertools.pairwise(cuts):
            if b - a < 5 * FS:
                continue
            x = ppg[a:b] - ppg[a:b].mean()
            y = signal.sosfiltfilt(sos, -x if self.invert else x)
            # scale by a 10 s running RMS so the prominence threshold follows the local amplitude
            k = int(10 * FS)
            rms = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same")) + 1e-9
            pk, _ = signal.find_peaks(y / rms, distance=max(1, int(self.min_distance_s * FS)), prominence=self.prominence_sd)
            beats.append(ppg_t_ms[a:b][pk])
        return {"beats_ms": np.concatenate(beats) if beats else np.zeros(0), "quality": None, "accept_windows": None}
