# Irregular-rhythm (AF) screen on beat intervals

Built and checked 2026-10-24. Code: `pulse_exploration/hrv/af_screen.py`; benchmark:
`benchmarks/af25/run.py`; tests: `tests/test_af_screen.py`.

## Rule

- Beats come from any source: ECG R-peaks, `foot_template` beats from 25 Hz PPG, or a cleaned
  interval list such as Samsung's IBIs.
- Intervals are cut into non-overlapping 64-beat segments of consecutive usable beats, as in the
  smartphone study by Lee et al. (IEEE TBME 2013).
- A segment is irregular when both hold:
  - RMSSD divided by the mean interval is at least 0.13, leaving out the 8 largest successive
    differences;
  - Shannon entropy of the intervals (8 longest and 8 shortest dropped, 16 bins, normalised) is at
    least 0.76.
  Both thresholds are Lee et al.'s, derived on the MIT-BIH AF and normal-rhythm databases.
- An alert needs 5 irregular segments among 6 consecutive ones, like the persistence rules of the
  cleared watch features. A positive screen calls for a 30 s ECG.

## Two findings that shaped it

**HRV cleaning hides AF.** `foot_template`'s HRV settings reject intervals more than 30 % off their
neighbours as artefacts. On synthetic pulse waves with AF beat timing, sampled at 25 Hz, that left
no 64-beat segment at all, so AF could never be flagged. The screen therefore uses
`AF_SETTINGS` (interval-based artifact correction off; beat-shape, motion and lost-sample handling
kept). With them all 68 synthetic AF segments were flagged and none of 60 sinus segments.

**A few bad beats look like AF unless trimmed.** Without interval cleaning, a few missed, extra or
mistimed beats per segment pushed 9.5 % of GalaxyPPG's sinus-rhythm segments over the threshold.
Leaving the 8 largest successive differences out of the RMSSD brought that to 0 %, while synthetic
AF stayed 100 % flagged (median 0.21, against 0.13).

## False alerts on GalaxyPPG

GalaxyPPG's 23 watch participants are healthy 20–29-year-olds in sinus rhythm, so every irregular
segment is a false positive.

| Beat source | People | Segments | Hours | Irregular segments | Alerts |
|---|---|---|---|---|---|
| Chest-strap ECG | 22 | 1,331 | 16.5 | 0.08 % | 0 |
| 25 Hz PPG, screen settings | 23 | 696 | 9.3 | 0 % | 0 |
| 25 Hz PPG, HRV settings | 23 | 582 | 7.8 | 0 % | 0 |
| Samsung IBIs, cleaned | 5 | 12 | 0.1 | 0 % | 0 |

Samsung's list rarely yields 64 consecutive adjacent intervals, because its one-second batches
break adjacency, so it is a poor source for this rule. The 95th percentile of the trimmed
normalised RMSSD in sinus rhythm is about 0.05, well below 0.13.

## Real AF: MIT-BIH

`benchmarks/afdb/run.py` uses PhysioNet's MIT-BIH Atrial Fibrillation Database (25 records of
about 10 h; 04936 and 05091 excluded for incorrect rhythm labels, as in Lee et al.) and Normal
Sinus Rhythm Database (18 records of about 24 h), both under the Open Data Commons Attribution
License v1.0. Only annotation and header files are downloaded, into `data/external/physionet/`.
Beat times come from the AF database's automatic, uncorrected QRS detections (manually corrected
ones where they exist) and from the normal-rhythm database's beat annotations, ectopic beats
included. Segments entirely inside AF count as AF, segments entirely inside normal rhythm as
sinus; mixed segments and atrial flutter are left out (6,705 AF and 26,318 healthy sinus
segments). Each rule also ran with 5 and 10 ms of Gaussian jitter on the beat times, to mimic
25 Hz PPG.

Per 64-beat segment, 5 ms jitter:

| Rule | Sensitivity | Specificity, AF patients' sinus rhythm | Specificity, healthy records |
|---|---|---|---|
| Lee et al. (untrimmed, 0.13) | 99.2% | 77.2% | 97.9% |
| Trimmed, 0.13 (default) | 87.6% | 94.4% | 99.7% |
| Trimmed, threshold chosen on the other records | 95.9% | 89.9% | 98.2% |

The leave-one-record-out choice settled on 0.09–0.11 (median 0.09). Trimming costs sensitivity
mostly at 90–120 bpm, where real AF is more regular than the synthetic AF used earlier (81% of
those segments flagged without jitter).

Alerts (5 of 6 segments), 5 ms jitter:

| Rule | AF records alerted | AF time inside alerts | False alerts per day, AF patients in sinus | False alerts per day, healthy |
|---|---|---|---|---|
| Lee et al. (untrimmed, 0.13) | 20 of 23 | 99% | 16.3 | 0.96 |
| Trimmed, 0.13 (default) | 21 of 23 | 85% | 3.1 | 0.13 |
| Trimmed, 0.09 | 21 of 23 | 97% | 7.5 | 1.22 |

The two records never alerted had only 1.9 and 2.8 minutes of AF, less than the roughly six
minutes the persistence rule needs. So the default suits screening (about one false alert per
week of continuous wear in healthy people); 0.09 suits tracking how much time someone with known
AF spends in it. False alerts in AF patients' sinus rhythm are frequent under every rule, and some
may be real arrhythmia the annotations do not label as AF.

## What is not known

- **Wrist PPG with AF.** MIT-BIH has ECG beat times; jitter mimics 25 Hz timing but not missed
  beats in a fast irregular rhythm or motion. DeepBeat (wrist PPG, Synapse) or the Vilnius wrist
  set would test that.
- **Older people and frequent ectopy.** Premature beats drive false alerts; the normal-rhythm
  database includes some, GalaxyPPG's young cohort almost none.
- **Our own nights.** The gate in the build plan is the false-alert rate on our own overnight
  recordings, which need all-night logging to work first.
