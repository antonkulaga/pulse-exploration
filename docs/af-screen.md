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

## What is not known

- **Sensitivity on real AF.** GalaxyPPG has no AF, and synthetic AF (independent intervals with a
  22 % spread) is easier than real AF at fast ventricular rates, which can be more regular. The
  trimming lowers AF's statistic by about a quarter and the threshold was derived without it.
  The next check is to run the rule on the MIT-BIH AF and normal-rhythm RR series (PhysioNet, open
  access), then on wrist PPG with AF (DeepBeat or the Vilnius set).
- **Older people and ectopic beats.** Frequent premature beats are the main source of false alerts
  in practice and are rare in GalaxyPPG's young cohort.
- **Our own nights.** The gate in the build plan is the false-alert rate on our own overnight
  recordings, which need all-night logging to work first.
