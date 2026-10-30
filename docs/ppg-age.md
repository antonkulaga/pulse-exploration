# Exploratory PPG age (AI-PPG Age)

Built and checked 2026-10-27. Code: `pulse_exploration/ppg_age.py` (beat averaging, model, CLI),
`pulse_exploration/net1d.py` (network); the optional `ppg_age` metric in
`watch_session.analyze_run`; tests: `tests/test_ppg_age.py`.

A PPG age is **never an absolute or biological age**. Per the project's decision it is shown only
with its error band and a warning, and is read only as a within-person trend. Every value the code
returns carries:

- the band (± 7.57 y, the authors' UK Biobank hold-out MAE, and ± 11.5 y zero-shot);
- the warning;
- the domain (wrist green at 100 Hz, or fingertip camera, against UK Biobank finger infrared);
- the number of beats averaged.

## Model

AI-PPG Age (Nie, Zhao, Tang, Li, Hong; [arXiv 2502.12990](https://arxiv.org/abs/2502.12990),
Communications Medicine 2025). This is the only open PPG clock with outcome evidence: in UK Biobank,
a gap above 9 years carried a MACCE hazard ratio of 2.37.

Checked on the [model card and files](https://huggingface.co/Ngks03/PPG-VascularAge):

- **Licence and access.** MIT (card front matter), ungated. The weights download without a token;
  the code sends none (`token=False`).
- **Files.** `model.pth` (1.6 MB), `config.json` and `net1d.py`, pinned at commit `611d2e2`.
  `model.pth` is checked by SHA-256 (`14589db1…`).
- **Input.** Shape (batch, 1, 100): one averaged beat, z-scored along its samples (the card's
  `normalize_ppg`). The training data is UK Biobank field 4205, the PulseTrace PCA2 finger infrared
  pulse, "100 sample points per waveform". The paper's Fig. 6 shows that this beat runs foot to
  foot, upright, with the systolic peak near sample 20 and the diastolic wave at 40–50.
- **Output.** One number, age in years.
- **Error.** MAE 7.57 y (r = 0.49) on the UK Biobank hold-out set. Zero-shot on VitalDB finger PPG
  ([Brag, arXiv 2603.14030](https://arxiv.org/abs/2603.14030)), the MAE was 11.50 y, and predictions
  collapsed into 38–67 y whatever the true age.

How it runs here:

- `net1d.py` is re-implemented in `pulse_exploration/net1d.py` with identical parameter names. Only
  the weights are downloaded; they load with `torch.load(weights_only=True)` and `strict=True`, and
  give 389,689 parameters, the paper's count. No downloaded code is executed.
- The card's remote inference API, on a raw IP address, is never called.
- torch comes from PyTorch's CPU wheel index (pinned in `pyproject.toml`). The model is tiny, and the
  CUDA build would add several GB.
- Fetch the weights once with `uv run python -m pulse_exploration.ppg_age --download`.
  `analyze_run` never downloads: without the weights, the metric is skipped with that instruction as
  its reason.

## Average beat

The beats are watch_session's own: `_ppg_input`, then `_ppg_beats` (template matching, NeuroKit2
fixpeaks), then `_ppg_intervals`. An interval is used when both of its beats match the template,
it is plausible and near its local median, and it lies wholly inside motion-free accelerometer
windows (1 s windows, 1 s padding).

1. Each used interval is cut from one beat fiducial to the next, on the 0.5–12 Hz pulse.
2. It is time-normalised to 100 samples with a cubic spline, linearly detrended so the cycle ends
   where it began, and z-scored.
3. The cycles are averaged.
4. The average is rotated to start at the foot: the minimum within 35 % of a cycle before the
   steepest upstroke.
5. Cycles that correlate below 0.8 with the average are dropped, and the rest are averaged again.
6. A 200-fold bootstrap over the cycles gives `resample_sd_y`. This is the spread from beat sampling
   within one recording, not accuracy.

Foot alignment matters. Shifting a beat's start by 5 samples moves the model's output by about
3 years.

The fingertip camera path (`camera_estimate`, CLI on a `data/camera/<id>-finger-trace` directory)
uses camera.py's primary channel (inverted red, or green when red is clipped). It has no
accelerometer, so movement shows only as beats that do not match their average.

## Model sanity checks

Synthetic beats, z-scored, foot at sample 0:

| Beat | AI-PPG Age |
|---|---|
| UK-Biobank-like, distinct late diastolic wave | 43.4 y |
| UK-Biobank-like, smaller and earlier diastolic wave | 59.1 y |
| UK-Biobank-like, no diastolic wave, peaks close | 67.9 y |
| Pure sinusoid (by phase) | 59.9–66.9 y |
| White-noise "beats" | 36–52 y (median 44) |

The model reads pulse shape in the paper's direction: a fading, earlier diastolic wave reads as
older. A beat with no shape lands at 60–67 y, the top of the zero-shot collapse range. So a beat that
is nearly a sinusoid (r ≥ 0.95 with one; UK-Biobank-like beats are 0.75–0.84) is marked unreliable.

## Reliability

The value is "experimental" at best, never "good". It is "unreliable" in any of these cases:

- there is no accelerometer;
- fewer than 10 motion-free intervals (the average then falls back to template-matched beats
  recorded during motion);
- the median beat-to-average correlation is below 0.9;
- the average beat is a near-sinusoid.

## Result on the existing run

The run is `data/runs/20261022-015808-record` (SM-R965F): 30 s of `sdk_ppg100` at 100 Hz, plus the
accelerometer. The wrist moved for 26 of the 30 s (median 1.2 m/s² against ~0.06 for a still
wrist), so after padding no second was still.

| | |
|---|---|
| Beats detected / template-matched intervals / motion-free intervals | 34 / 19 / 0 |
| Beats averaged | 17 (2 dropped against the average); median correlation 0.94 |
| Average beat | peak at sample 37, no diastolic wave; sinusoid correlation **0.97** |
| AI-PPG Age | **68.8 y**, band 61.3–76.4 y (zero-shot band 57.3–80.3 y) |
| Beat-resample SD | 6.3 y |
| Reliability | **unreliable**: no motion-free beats; near-sinusoid beat |

This number carries no information about age:

- The beat is the shapeless kind the model reads as 60–67 y.
- 68.8 y sits just above the 38–67 y range where zero-shot predictions collapse.
- Resampling the same 17 beats alone moves it by ±6 y.

It must not be compared with a calendar age.

There is no real fingertip result yet. The three `finger` traces in `data/camera` were recorded on
the Android emulator: a synthetic scene at exactly 120 bpm. They served only as a smoke test; all
three came out as near-sinusoids, unreliable.

Reproduce it (this prints the value, the gate and the warning):

```bash
uv run python -m pulse_exploration.ppg_age data/runs/20261022-015808-record
```

## What the next recordings need

**Repeated `spot ppg100` checks at a fixed posture.** A check is 30 s, the on-demand maximum. Use the
watch button "PPG 100 Hz 30 s", or:

```bash
$A shell am start -n $PKG/.MainActivity --es cmd spot --es feature ppg100 --ei seconds 30
```

- Posture: seated after 5 minutes' rest, forearm resting on a table at heart level, watch snug on the
  same wrist and position, no talking, the same time of day.
- Each check must be still. The gate needs 10 motion-free intervals and its correlation checks;
  aim for 25 or more beats.

**About 10 checks per quarter**, the OPPO smartwatch heart-age protocol (repeated user-initiated 30 s
spot checks, at least 10 per person over 3 months; arXiv 2608.01620).

- Report the quarter's median PPG age with the spread of its checks, as a within-person trend.
- A change counts only if it is larger than the spread within a quarter.
- The trend itself is not validated on this device. The UK Biobank gap-to-outcome evidence does
  not transfer to the wrist.

**Pair some checks with a fingertip camera capture** (capture page, torch, locked exposure,
60–90 s). The fingertip is closer to the model's training domain. Wrist and finger values that
move together over time would be the first sign that the wrist value tracks anything.
