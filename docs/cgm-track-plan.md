# CGM-paired research track: plan for testing with Livia

Drafted 2026-10-24. Nothing here is built yet. It is the labelled research track from the build
plan in the [research report](../reports/PPG%20health%20and%20aging%20measures.md), turned into
steps now that a CGM wearer can test it.

## Question

Do overnight watch signals (heart rate, HRV, motion, skin temperature) carry information about
glucose events beyond what the time of night, recent insulin and recent carbs already predict?
With a CGM on, the CGM itself detects these events, so the watch's possible value is an earlier
warning, a signal while the CGM is warming up or has gaps, and later a screen for people without a
CGM.

Targets, in order:
1. **Nocturnal low glucose:** below 70 mg/dL, and below 54. Lows trigger an autonomic response
   (faster heart rate, lower HRV, sweating) that wrist signals can see; smartwatch data detected
   them with AUROC 0.76 in 22 insulin-treated adults (Lehmann et al., Diabetes Care 2023).
2. **After-meal rises:** size and speed of the rise against heart rate and HRV.
3. **Day-level links:** glucose variability against the next night's HRV and resting heart rate.

Published work warns that clock time and food intake explain most of what wristbands seem to
predict (Bent et al., npj Digital Medicine 2021), so every result is judged against those
baselines.

## Data, and where each part comes from

| Data | Source | Status |
|---|---|---|
| Six years of glucose history (Dexcom G6, 5-min grid, Oct 2019 – Sep 2025) with fast- and long-acting insulin | `~/data/Livia/OUTPUT/livia_glucose_insulin.parquet`, built with glucose_data_processing; never committed | Exists; ends before any watch data, so it serves for building the loader, labels and baselines, not for pairing |
| Glucose every 5 min, carbs, bolus and basal insulin, insulin on board, going forward | GlucoWatch's CSV export (one event per row; `time_millis` is the join key; documented in GlucoWatch `docs/export.md`) or new Clarity exports | Exists |
| Overnight 25 Hz PPG and accelerometer | Our watch probe app, all-night session | Needs all-night logging to work (Phase 0) |
| Heart rate as a fallback | Samsung Health through Health Connect; GlucoPhone already reads it | Exists, minute-level, synced in batches |
| Skin temperature | Samsung Health Data SDK, or the SDK tier on the watch | Needs one of the two |

Questions for Livia before starting: which watch, whether Samsung Health is in use, which CGM
and insulin data GlucoWatch reads today, and how meals will be logged: the historical file has no
carb entries in its last 12 months (2,650 in earlier years).

## What to build

1. **CGM loader**: GlucoWatch CSV to a 5-minute grid (glucose from one source, carbs and insulin
   as events, insulin on board forward-filled only within a sample's validity, as GlucoWatch's
   docs require).
2. **Night features** from probe-app sessions, per 5 minutes: mean and minimum heart rate, RMSSD
   from `foot_template` on motion-free beats, movement, skin temperature when present.
3. **Event labels** from the CGM: onset of a low (first reading under 70 after at least 30 minutes
   above), level-2 lows (under 54), fast rises. The CGM lags blood glucose by roughly 5–10
   minutes, which matters for any lead-time claim.
4. **Evaluation**: leave-one-night-out. Compare a watch model against baselines using only the time
   of night, insulin on board and carbs, and report how many minutes before the CGM threshold
   crossing any warning comes.
5. **Weekly report** for Livia: nights recorded, events, what the watch showed around each.

Steps 1, 3 and 4 can be built and tested on the historical glucose file before Livia's first
night; step 2 waits for all-night logging.

## How long

In the last 12 months of the historical file (Sep 2024 – Sep 2025, 363 nights), lows between
midnight and 6 am lasting at least 15 minutes happened about 11 times per 30 nights below
70 mg/dL and about 4 times below 54. Twenty events below 70 therefore take about two months of
recorded nights, and 50 about four and a half. After-meal analysis needs meal times first.

## What Livia would do

- Wear the watch overnight with the probe app (or at least Samsung Health heart rate).
- Export the GlucoWatch CSV once a week.
- Keep logging meals and boluses as usual.

The data stays local and is Livia's health data; nothing here is a medical device or a basis for
treatment decisions.
