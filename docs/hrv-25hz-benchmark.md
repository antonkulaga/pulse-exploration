# Can the watch's 25 Hz PPG give HRV without the Samsung SDK?

Checked 2026-10-22 on the public GalaxyPPG dataset: four 25 Hz beat-timing methods and Samsung's own
beat-interval list, each scored window by window against a chest-strap ECG. Methods were tuned on 5
participants and scored on 17 others they had never seen.

## Short answer

- **Yes, at rest and during light, seated activity.** The two best methods (`foot_template`,
  `motion_denoise`) estimate 60 s RMSSD with a median error of about 3 ms (bias +2 ms, 95% limits
  −8…+12 ms). At seated rest they report 88–94% of the windows, and 98% of those are within ±10 ms
  of the ECG. Over 5-minute windows, the clinical standard, `foot_template` reported all 34 and
  every one was within ±10 ms.
- **Not while walking or running.** Every method except `naive` declines all locomotion windows;
  `naive` reports them and is always wrong. Typing is marginal: about half the windows are reported,
  70% of those within ±10 ms.
- **Naive processing fails, as in the paper.** Band-pass and peak picking overestimates RMSSD by
  about 180 ms. Precise beat timing between samples, beat-shape checks and artifact handling are
  what make it work.
- **Samsung's SDK interval list needs the same cleaning.** Taken raw it is worse (median error
  10 ms, bias +32 ms, half its windows within ±10 ms), because it contains unflagged artifacts and
  the same lost-sample errors as the PPG. With the cleaning the 25 Hz methods get, it is about as
  good: 4.4 ms, bias +1.5 ms, 91% within ±10 ms. So the SDK adds no accuracy for RMSSD at rest; it
  is a second, independent estimate.
- **Not yet shown on your watch.** GalaxyPPG was recorded on a Galaxy Watch5 through the Samsung
  SDK's green-PPG tracker. Our no-SDK source is the `hr_raw` sensor on a Watch6 Classic, which
  carries the same pulse (r ≈ 0.95) but may time beats differently. One 30 s ECG + `hr_raw`
  recording on your watch settles that.

## Data

GalaxyPPG (Park, Zheng & Lee, *Scientific Data* 12:892, 2025; Zenodo 10.5281/zenodo.14635823,
CC-BY-4.0), stored in the gitignored `data/external/galaxyppg/`.

- 24 participants aged 20–29, 12 women and 12 men, about an hour each.
- Galaxy Watch5: 25 Hz green PPG, 25 Hz accelerometer, and Samsung's 1 Hz heart rate with its
  beat-interval (IBI) lists.
- Polar H10 chest strap: ECG at about 130 Hz, and Polar's own RR intervals.
- Labelled sessions: adaptation, baseline, meditation, rest, screen reading, stress tasks (TSST
  preparation and speech), typing on a keyboard and on a phone, standing, walking, jogging, running.

P01 has no watch data. P11's ECG is too noisy to serve as a reference, so P11 drops out. P07 and
P08 have no Samsung heart-rate file, so Samsung's IBI list is scored on 15 held-out participants,
the PPG methods on 17.

## Method

**Reference beats.** R-peaks from the ECG with NeuroKit2 at the ECG's real rate, refined between
samples, with Kubios artifact correction. A beat is invalid if Kubios moved or added it, if its shape
correlates below 0.6 with its neighbours, or if its 10 s segment fails NeuroKit2's quality check. An
interval counts as normal-to-normal (NN) only if both beats are valid, it lies within 300–2000 ms
and within ±30% of the local median. For most participants 97–100% of NN intervals agree with
Polar's own RR within 10 ms (P15: 90%), at a mean difference under 1 ms for most.

**Clock alignment.** The watch and the strap have separate clocks. One fixed method maps ECG beats
onto the watch clock for every participant: an R-peak-triggered average of the PPG upstroke, fitted
in 5-minute chunks for drift. It uses no candidate's output. RMSSD and SDNN do not depend on the
alignment beyond which beats fall inside a window.

**Windows.** Non-overlapping 60 s and 300 s windows inside each session. A window is scored when at
least 90% of its reference intervals are NN, those agree with Polar, and at least 90% of the PPG
samples are present. 681 held-out 60 s windows qualify; 134 fail the NN rule. Only the adaptation,
meditation and some rest sessions are long enough for 300 s windows (34 held out).

**Coverage.** A method may decline a window, from the PPG and accelerometer alone. A window is
*covered* when the method accepts it and returns enough intervals (at least 10 and at least half the
reference count). Errors are computed on covered windows; *yield* is the share of all scored windows
that are covered and within ±10 ms, so declining does not hide failure.

**Fairness.** Methods were tuned only on P02–P06 and frozen before the held-out run on P07–P24.
`benchmarks/hrv25/run.py` regenerates everything from the dataset in about 15 s on 20 cores.

## The methods

| Method | What it does |
|---|---|
| `naive` | 0.5–4 Hz band-pass, `find_peaks` on the 25 Hz samples. The trivial baseline. |
| `classical` | Elgendi systolic-peak detection on the inverted signal, then timing at the half-amplitude point of the upstroke on a 1 kHz spline, a shift to match the running median beat, Kubios correction, and a beat-shape and accelerometer gate. |
| `foot_template` | Steepest upstroke on a 1 kHz spline, aligned by cross-correlation to the median beat of its 31 neighbours; beat-shape, motion and interval-artifact rejection; corrects the stream's lost samples (below). |
| `motion_denoise` | Accelerometer regression removes motion from the PPG when the wrist moves; beats tracked by dynamic programming on an autocorrelation period estimate; steepest upstroke on a ×10 upsampled signal; artifact and lost-sample handling; declines sustained heavy motion. |
| `samsung_ibi` | Samsung's IBI list from the SDK heart-rate tracker (`ibiStatus` 0 only). Intervals only, so no beat-level scores. Samsung reports each beat about 7.4 s after it happens (median over P02–P06; 6.3–8.5 s per participant), so the list is shifted back by that constant. |
| `samsung_ibi_clean` | The same list after the cleaning the 25 Hz methods do for themselves: intervals outside 300–2000 ms or more than 20% from the median of 11 neighbours are dropped, and so are isolated intervals more than 30 ms shorter than both neighbours' mean (lost samples). |

The authors of `foot_template` and `motion_denoise` were cut off by a usage limit before they wrote
their reports; their code and tests were complete, and their defaults at that point are what was
scored.

## Results on the held-out participants (P07–P24)

60 s windows, all activities:

| Method | Covered | RMSSD median error | Bias (95% limits) | Within ±10 ms | Yield | SDNN median error | IBI error |
|---|---|---|---|---|---|---|---|
| naive | 99% | 103 ms | +182 (−233…+597) | 3% | 3% | 55 ms | 17.8 ms |
| classical | 63% | 5.2 ms | +5.5 (−5…+16) | 91% | 58% | 2.5 ms | 7.1 ms |
| foot_template | 79% | 3.1 ms | +2.5 (−6…+11) | 95% | 75% | 2.1 ms | 7.1 ms |
| motion_denoise | 75% | 3.0 ms | +1.9 (−8…+12) | 95% | 71% | 2.3 ms | 8.0 ms |
| samsung_ibi (SDK) | 77% | 10.0 ms | +32 (−68…+133) | 50% | 38% | 5.0 ms | — |
| samsung_ibi_clean (SDK) | 72% | 4.4 ms | +1.5 (−11…+14) | 91% | 65% | 2.7 ms | — |

The median reference RMSSD in these windows is about 25 ms, so 3 ms is about 12%. On the windows
every method could be scored on (615, excluding P07/P08), the numbers barely move. On the 377
windows that `classical`, `foot_template`, `motion_denoise` and raw Samsung all covered, the two best
methods are within ±10 ms in 98–99% of windows, against 61% for raw Samsung.

300 s windows (seated rest only, 34 windows from 14 participants):

| Method | Covered | RMSSD median error | Bias (95% limits) | Within ±10 ms |
|---|---|---|---|---|
| naive | 100% | 28.5 ms | +86 (−148…+320) | 0% |
| classical | 94% | 4.5 ms | +4.6 (−4…+14) | 97% |
| foot_template | 100% | 2.4 ms | +2.1 (−4…+8) | 100% |
| motion_denoise | 97% | 2.4 ms | +0.9 (−6…+8) | 97% |
| samsung_ibi (SDK) | 97% | 9.0 ms | +26 (−66…+118) | 55% |
| samsung_ibi_clean (SDK) | 97% | 3.7 ms | +1.0 (−9…+11) | 97% |

By activity, 60 s windows. Each cell: share of scored windows covered / median RMSSD error /
share of covered windows within ±10 ms.

| Activity (windows) | classical | foot_template | motion_denoise | samsung_ibi (SDK) | samsung_ibi_clean (SDK) |
|---|---|---|---|---|---|
| Seated rest (377) | 82% / 4.8 ms / 94% | 94% / 2.7 ms / 98% | 88% / 2.6 ms / 98% | 93% / 8.7 ms / 58% | 87% / 3.9 ms / 93% |
| Cognitive stress (128) | 63% / 5.7 ms / 89% | 84% / 3.3 ms / 96% | 84% / 3.4 ms / 96% | 88% / 11.3 ms / 38% | 82% / 5.0 ms / 89% |
| Standing (39) | 74% / 7.6 ms / 83% | 92% / 5.5 ms / 86% | 90% / 4.7 ms / 91% | 63% / 9.6 ms / 55% | 63% / 5.4 ms / 91% |
| Typing (79) | 15% / 11.0 ms / 42% | 49% / 5.0 ms / 72% | 46% / 6.6 ms / 67% | 46% / 56 ms / 6% | 38% / 7.3 ms / 74% |
| Walking, jogging, running (58) | declined | declined | declined | declined | declined |

Samsung's denominators are smaller (344, 114, 35, 71, 53) because P07 and P08 have no heart-rate file.

Per session, `foot_template` covers 98–100% of adaptation, baseline, meditation and screen-reading
windows at 2.1–3.3 ms; 82% of the between-exercise rest windows (4.3 ms, heart rate still
recovering); 66% of TSST speech (3.1 ms); 90% of phone typing (4.5 ms) but 8% of keyboard typing.
With both of the best methods, no held-out participant's median error exceeds 5.1 ms.

**Where the best methods are weakest.** In windows whose ECG RMSSD is 15 ms or less (186 of 681,
mostly standing and stress), coverage drops to about 62% and the bias rises to +4 to +5 ms, which is
+30–40% relative. Above 50 ms (40 windows), `motion_denoise` reads 4 ms low and covers 65%, because
its lost-sample rule also trims genuine short intervals; `foot_template` holds 82% coverage there.
Cleaned Samsung is weakest at high HRV: −10 ms bias over 21 windows.

The development set (P02–P06) gave the same picture: 3.5–3.7 ms for the two best methods, 5.5 ms for
`classical`, 11.6 ms for raw Samsung and 3.5 ms cleaned. Held-out numbers are slightly better, not
worse, so there is no sign of overfitting.

Plots: `data/benchmarks/hrv25/held_out_bland_altman_60s.png` (and `_300s`), and
`held_out_by_group_60s.png`.

## A data finding: the watch silently drops samples

Two methods found this independently. About once every 10–12 s the 25 Hz stream loses one sample
while its timestamps stay evenly spaced. Every beat after the loss is then stamped 40 ms early, and
one interval comes out about 40 ms short. That single short interval inflates RMSSD noticeably.
`motion_denoise` voids such intervals; `foot_template` finds the chain of losses and undoes it.

The review confirmed it independently. The PPG timestamps are spaced 40.0 ms apart at the median
but 40.15 ms on average, which is one lost sample every 10–11 s. Against the ECG, 7–9% of
uncorrected intervals are 25–55 ms too short and only 1–3% that much too long. Samsung's own IBI
list shows the same skew, so its beat detector runs on the same stream. Switching the correction
off on the held-out set raises the RMSSD error to 7.2 ms (`foot_template`) and 6.2 ms
(`motion_denoise`). Neither correction is perfect: `foot_template` adds a few +40 ms errors, and
`motion_denoise` trims genuine short intervals at high HRV. Whether our own `hr_raw` stream loses
samples the same way is unknown; the watch's ECG will tell.

## Samsung's interval list needs the same cleaning

Raw, its errors are mostly small with a tail of large overestimates. Two causes, both found by the
review:

- **Unflagged artifacts.** Some intervals with `ibiStatus` 0 are plainly wrong (500 ms next to
  2300 ms). The status itself is meaningful: intervals marked −1 agree with the ECG within 10 ms
  only about 20% of the time.
- **Lost samples.** The same short-interval errors as in the PPG.

The scoring itself is fair: 99.7% of interval pairs the harness treats as consecutive are
consecutive ECG beats, units are milliseconds, and interval sums match the batch spans within 0.1%.
With the generic ±20% local-median filter alone, Samsung improves to 8.4 ms (bias +9.8); with the
lost-sample rule as well it reaches the 4.4 ms of `samsung_ibi_clean`.

## How the benchmark was checked

An independent review re-read the harness and candidates and re-ran parts with its own code:

- **Leakage: none found.** Candidates import only numpy, scipy and NeuroKit2, receive only PPG and
  accelerometer data, and contain no participant-specific logic. Their tuning runs cover P02–P06
  only. Clock alignment and the per-window delay affect beat matching, never RMSSD.
- **Reference: sound.** ECG at 130.3–130.6 Hz; NN intervals agree with Polar's RR with a mean
  difference under 1 ms for most participants (worst 4 ms, P07 and P19).
- **Same windows for everyone.** Usable windows and reference values are identical for all beat
  methods; Samsung differs only by P07/P08 and by 2 windows where the PPG-gap rule applies.
- **Recomputation.** RMSSD for six random held-out windows, recomputed from the beats with separate
  code, matches the tables to 3 decimals for `foot_template` and for the ECG.
- **Fixed during the check:** intervals spanning a PPG recording gap were counted (up to 850 ms
  of RMSSD error in one window; now excluded, with a test), and Samsung's IBIs were placed 7.4 s
  late (now shifted, which changed its numbers only slightly).

## Limitations

- **Different watch and source.** Watch5 SDK green PPG here, Watch6 Classic `hr_raw` without the SDK
  for us. The decisive check is a simultaneous watch ECG and `hr_raw` recording.
- **Young, healthy, semi-naturalistic.** 20–29-year-olds in a lab protocol. No older people, no
  arrhythmia, no sleep. Night recordings are likely easier (still), but untested.
- **RMSSD and SDNN only.** Frequency-domain HRV and pNN50 were not evaluated for decisions; pNN50
  error is in the CSVs.
- **Coverage is the method's own choice.** That is how the app would work, and yield accounts for
  it, but a method that declines more looks more accurate per window (compare `classical`).
- **17 held-out participants.** Enough to rule out the paper's failure mode, not to set tight
  population limits.

## What the server does with this (analysis version 0.3)

`pulse_exploration/watch_session.py`, which serves `/runs/<run>/analysis`, now uses these results.

- **25 Hz PPG HRV (`sdk_ppg`, `hr_raw`) comes from `foot_template`**, with its own beat-shape,
  motion and artifact checks. A run is cut into windows of about 60 s, spanning the whole recording
  so off-wrist time counts against them. A window counts when at least half its expected intervals
  survive (and at least 10), as in the benchmark; HRV uses the intervals of the windows that count.
- **Labels.** `sdk_ppg`, the source the benchmark validated, is "good" with at least 54 s of
  accepted windows. `hr_raw` stays "experimental" until it has been checked against the watch ECG
  (`HRV25_BEST`). Shorter accepted stretches are "experimental", and no accepted window is
  "unreliable".
- **Per-axis motion gate.** `foot_template`'s gate looks at the acceleration magnitude, which a
  wrist turn barely changes. On synthetic runs that let badly timed beats through (timing error SD
  270 ms). The server adds a per-axis gate (`axis_max` = 1.0 m/s² over 2 s), chosen on P02–P06
  where 95% of seated-rest seconds stay below it. On the held-out set it costs 2 points of coverage
  at the same error (3.2 vs 3.1 ms). The accelerometer is averaged to 25 Hz first, because
  `foot_template` assumes 25 Hz and the phone's Android accelerometer runs at 100 Hz.
- **Samsung's IBI list** is cleaned as `samsung_ibi_clean`: outliers against the median of 11, and
  isolated intervals more than 30 ms shorter than both neighbours' mean. Its motion check is moved
  6 s back, because Samsung lists each beat about 6 s after it happened. On your SM-R965F the list
  matched `sdk_ppg` beats 5.2–7.0 s earlier, and the first rows of a run list beats from before the
  recording.
- **HRV priority** is now ECG, `sdk_ppg`, Samsung IBIs, 100 Hz PPG, `hr_raw`. When `sdk_ppg` (or
  `hr_raw`) and Samsung's IBIs are both usable and more than 10 ms apart, the analysis warns
  (`hrv_sources_disagree`).
- **Motion threshold** for the rest of the analysis (`MOTION_RMS`) went from 0.2 to 0.5 m/s², chosen
  on GalaxyPPG P02–P06: it keeps 92% of seated-rest and 96% of stress-task seconds and no walking or
  running second. 0.2, tuned on one of your recordings, called a still run 36% motion.

**High HRV is the weak spot of both lost-sample corrections.** On ECG beats, which have no lost
samples, `foot_template`'s correction finds 6–23 false losses per hour at RMSSD around 20 ms but
250 per hour at 46 ms, close to the real rate of about 330. Samsung's rule drops 1% of genuine
intervals at RMSSD 20 ms and 11% at 46 ms. Your own 30 s recording is in that regime: `sdk_ppg`
59 ms and `hr_raw` 63 ms agree, but cleaned Samsung gives 36 ms (69 ms without the lost-sample
rule). Both paths add a note above 50 ms or when the rule drops more than 10% of intervals. Gating
the rule on how calm the neighbouring intervals are was tried and rejected: on P02–P06 it fixed the
high-HRV bias (−7.6 → +2.8 ms) but worsened everyone else (3.5 → 5.4 ms).

## Reproduce

```bash
uv run python benchmarks/hrv25/run.py --workers 20
uv run pytest -q tests/test_hrv_bench.py tests/test_cand_classical.py tests/test_cand_foot_template.py tests/test_cand_motion_denoise.py
```

Results go to `data/benchmarks/hrv25/` (gitignored): `windows.parquet` (one row per method and
window), summary CSVs per split (`overall`, `by_group`, `by_session`, `by_participant`, `within`,
and `*_common` on shared windows), and the plots.
