# pulse-exploration

A research prototype that measures heart signals with devices people already own: a Samsung Galaxy
Watch (developed on a Watch6 Classic, SM-R965F) and a phone camera. A laptop server analyses the
recordings. It is not a medical device and was built without regulatory work. Recordings are
personal health data and stay out of git.

The project started from one watch, the author's own, to keep things simple. The longer-term aim is
aging biomarkers: pulse-based measures read with existing PPG age models and, later, combined with
methylation, blood and photo clocks. See [Aging clocks](#aging-clocks-where-this-is-heading) below.

The guiding question is what can be measured **zero-shot**, without labelled training data of our
own. Every feature is tagged by what the user has to set up:

- **tier `25hz`:** plain Android sensors; installing the app is enough.
- **tier `all`:** the Samsung Health Sensor SDK, which needs Health Platform developer mode on the
  watch (or Samsung partner approval).

The apps use the SDK when the watch allows it and fall back to tier `25hz` otherwise.

## What was done

Work ran from 2026-10-21 to 2026-10-27. Each piece has a document with methods, full tables and
limitations; the numbers below are the headlines.

### 1. Raw signals from the watch without the Samsung SDK

- The Wear OS app [`watch/probe`](watch/probe) records sensors in a foreground service and uploads
  each run to the laptop server. At start-up it probes which sources the watch allows and writes
  them to `capabilities.json` in every run.
- Samsung's hidden `com.samsung.sensor.hr_raw` sensor opens through plain Android `SensorManager`
  with only the heart-rate permission. It gives 16 channels at about 25 Hz. Channel v5 matches the
  SDK's green PPG (r = 0.98) and is the one the server uses; v1/v4 and v9/v11/v13 are other light
  channels, and the rest are status, ambient or unused.
- With the SDK, the watch adds green/IR/red PPG at 25 Hz, 100 Hz on-demand PPG, 500 Hz single-lead
  ECG, Samsung's heart rate with beat-interval (IBI) lists, SpO2 and skin temperature.
- The 25 Hz streams silently drop about one sample every 10–12 s while their timestamps stay evenly
  spaced, so every later beat is stamped 40 ms early. `hr_raw` hides its losses this way; the SDK
  stream shows them as gaps. Uncorrected, this one short interval inflates HRV.

Details: [docs/phase0-offline-findings.md](docs/phase0-offline-findings.md),
[docs/model-feasibility.md](docs/model-feasibility.md).

### 2. HRV from 25 Hz watch PPG, benchmarked against ECG

On the public GalaxyPPG dataset (Galaxy Watch5 PPG with a Polar H10 chest-strap ECG, 24 people
aged 20–29), four 25 Hz beat-timing methods and Samsung's own IBI list were tuned on 5 participants
and scored on 17 others they had never seen.

| Method (60 s windows, held out) | RMSSD median error | Within ±10 ms of ECG |
|---|---|---|
| `naive` (band-pass + peak picking) | 103 ms | 3% |
| `classical` | 5.2 ms | 91% |
| `foot_template` (used by the server) | 3.1 ms | 95% |
| `motion_denoise` | 3.0 ms | 95% |
| Samsung IBIs, raw | 10.0 ms | 50% |
| Samsung IBIs, cleaned like the PPG | 4.4 ms | 91% |

- The 25 Hz PPG gives resting HRV about as well as the SDK's interval list, as long as beats are
  timed between samples, checked for shape and corrected for lost samples.
- At seated rest the best methods report 88–94% of windows; over 5-minute windows `foot_template`
  was within ±10 ms in all 34.
- No method gives usable HRV while walking or running; they decline those windows.

Details: [docs/hrv-25hz-benchmark.md](docs/hrv-25hz-benchmark.md).

### 3. Pulse shape survives 25 Hz when beats are averaged

Wrist green PPG holds almost no power above 12.5 Hz. Average beats rebuilt from 25 Hz samples
reproduced the 64 Hz reference's crest time within about 1 ms and the second-derivative b/a ratio
within 0.003. So the 25 Hz limit is not what stops pulse-shape (vascular) measures; device filtering,
lost samples and noise still might. Details:
[docs/phase0-offline-findings.md](docs/phase0-offline-findings.md).

### 4. Irregular-rhythm (AF) screen

A screen on beat intervals ([`pulse_exploration/hrv/af_screen.py`](pulse_exploration/hrv/af_screen.py)),
based on the RMSSD-plus-entropy rule of Lee et al. (2013), with two changes found while testing: it
skips the HRV cleaning, which hid AF entirely, and it trims the 8 largest successive differences,
which removed false alerts from a few bad beats.

- On PhysioNet's MIT-BIH AF and Normal Sinus Rhythm databases, with beat times jittered to mimic
  25 Hz PPG, it alerted in 21 of 23 AF records (the two missed had under 3 minutes of AF) at
  0.13 false alerts per day in healthy records, against 0.96 for the untrimmed published rule.
- On GalaxyPPG, where nobody has AF, it raised no alerts.
- Not yet tested on wrist PPG recorded during AF, or on older people with frequent ectopic beats.

Details: [docs/af-screen.md](docs/af-screen.md).

### 5. First check against the watch's own ECG

On 2026-10-26 the watch recorded its SDK ECG and `hr_raw` together for about 45 s.
[`benchmarks/watch_ecg/run.py`](benchmarks/watch_ecg/run.py) scores the server's 25 Hz beats
against the ECG R-peaks with a pass gate fixed before scoring. In that one recording, 37 of 38 beats
matched, beat timing SD was 14.5 ms and RMSSD came out 49.9 ms against 55.6 ms from the ECG; it
passed the gate. The same recording exposed two ECG parsing bugs in the server (lead-off flags and
late timestamps), which are fixed. One recording is not enough to promote `hr_raw`, so the server
still labels its HRV "experimental".

### 6. Phone-camera PPG

- The Android app [`watch/phone`](watch/phone) records fingertip video (rear camera, torch on) or a
  face region (front camera). It locks exposure and white balance and sends per-frame colour means,
  never video, to the server.
- [`pulse_exploration/camera.py`](pulse_exploration/camera.py) estimates heart rate:
  - fingertip: classical analysis of the inverted red/green signal;
  - face: POS/CHROM, plus the pretrained ME-rPPG model through `open-rppg`.
- Quality checks flag a missing fingertip, face signals too strong to be a pulse, and a deep-model
  heart rate that disagrees with the classical one by more than 5 bpm.
- The app has run only on the Android emulator, never on a real phone.
- The server also serves web pages (`/`, `/upload`, `/capture`) as a browser fallback.

See [watch/phone/README.md](watch/phone/README.md).

### 7. Research and planning

- [reports/PPG health and aging measures.md](reports/PPG%20health%20and%20aging%20measures.md): what
  wrist and camera PPG can and cannot measure, graded by evidence. Its conclusions:
  - nightly beat timing (resting heart rate, HRV, respiratory rate, rhythm, illness alerts) has the
    strongest evidence and works on tier `25hz`;
  - PPG aging clocks with good accuracy need population-scale training and long averaging, so
    ours can only be a within-person trend;
  - calibration-free blood pressure, glucose and lipids from PPG should not be built.
  
  It ends with a phased build plan. The notes it was synthesised from are in
  [research_notes/](research_notes/PPG%20health%20and%20aging%20measures).
- [docs/model-feasibility.md](docs/model-feasibility.md): which open models fit these devices.
  - PulseLM and the other Manhph2211 models are not drop-in zero-shot models for them.
  - Classical signal processing (NeuroKit2) is the baseline every model has to beat.
  - Better-suited open models: ECGFounder (single-lead ECG), Pulse-PPG (watch PPG), AnyPPG and
    ME-rPPG (camera).
- [docs/cgm-track-plan.md](docs/cgm-track-plan.md): a plan, not yet built, for an N-of-1 study pairing
  overnight watch signals with a continuous glucose monitor to look for nocturnal lows.

## Aging clocks: where this is heading

The plan is to use existing PPG age models rather than train a new one. Good PPG clocks need
population-scale data: Apple's PpgAge reached a 2.4-year error with 213,593 people and 30-day
averages. A prototype with a few users cannot match that, but it can feed such models clean,
repeated measurements of the same person.

**What this repository already gives an aging clock**

- **Beat timing.** Nightly resting heart rate and HRV have the strongest outcome evidence of any
  wrist measure (resting heart rate: all-cause mortality RR 1.09 per 10 bpm across 46 cohorts). The
  25 Hz HRV here is within about 3 ms of ECG.
- **Pulse shape.** Averaged beats from the 25 Hz stream keep crest time and the b/a ratio
  (`benchmarks/morph25`). The SDK's 30 s spot checks add 100 Hz PPG and an ECG for pulse arrival time.

**Existing models to use**

| Model | What it reads | Error | Outcome evidence | Access |
|---|---|---|---|---|
| AI-PPG Age (Nie et al., *Communications Medicine* 2025) | one averaged finger beat, 100 samples | 7.6 y in UK Biobank; 11.5 y on other data | gap above 9 y: death HR about 2.5 | MIT, [weights on Hugging Face](https://huggingface.co/Ngks03/PPG-VascularAge) |
| Pulse-PPG, PaPaGei, AnyPPG | embeddings of wrist, finger or phone PPG | 6–10 y with an age probe | none yet | Pulse-PPG MIT; the others need permission |
| Apple PpgAge, OPPO heart age | wrist PPG, 30-day averages or repeated 30 s checks | 2.4 y and 5 y | PpgAge gap predicts heart disease | closed; their methods shape our protocol |

AI-PPG Age is the only open PPG clock with outcome evidence, so it comes first. It runs in
[`pulse_exploration/ppg_age.py`](pulse_exploration/ppg_age.py); details in
[docs/ppg-age.md](docs/ppg-age.md).

- The weights are downloaded once from a pinned commit and checked by SHA-256. The network is
  re-implemented in `net1d.py`, so no downloaded code runs.
- Beats are averaged from motion-free intervals into the model's 100-sample, foot-to-foot input.
- Every value carries its error band, a warning and the number of beats averaged. It is marked
  unreliable with fewer than 10 motion-free intervals, beats that match their average poorly, or a
  shapeless, near-sinusoid average beat.
- The watch analysis adds it as an optional `ppg_age` metric for runs with 100 Hz PPG. Fingertip
  camera traces work through the command line.

On the first 30 s recording the wrist moved for 26 of 30 s. The result, 68.8 y, was marked
unreliable and says nothing about age; still recordings are the next step.

**How the numbers will be used**

- A PPG age from these devices is a within-person trend, always shown with its error band and a
  warning, never compared with calendar age.
- Measurements follow the OPPO protocol: about 10 still 30 s spot checks per quarter at a fixed
  posture, some paired with a fingertip camera capture, which is closer to AI-PPG Age's finger
  training data.
- A change counts only when it is larger than the spread of checks within a quarter.

**Later: combining with other clocks**

Pulse measures read the vascular and autonomic systems and change within days. Methylation clocks
(GrimAge2, DunedinPACE), blood clocks (PhenoAge, proteomic age) and photo clocks (FaceAge, retinal
age) read other systems over months to years. Ways to combine them, from cheapest to hardest:

- show each clock's age gap side by side, grouped by body system;
- fit them together against death and disease outcomes, for which UK Biobank holds finger pulse waves,
  accelerometry, blood chemistry, proteins and retinal photos;
- give cohorts that have methylation but no wearables a methylation or blood surrogate of the pulse-age
  gap;
- use yearly molecular clocks as anchors and the watch to track the months between them.

The [PPG health and aging report](reports/PPG%20health%20and%20aging%20measures.md) and its
[aging research notes](research_notes/PPG%20health%20and%20aging%20measures/aging_vascular.md) hold
the evidence. The [PPG Aging Roadmap](https://claude.ai/artifact/3J2bE9UCWxAzkoqtrHYfCw) (private
page) lays out the build order and open decisions.

## How the pieces fit

```text
Galaxy Watch (watch/probe) ──PUT /runs/{run}/{file}──▶ laptop server ──▶ data/runs/<run>/
                                                       GET /runs/{run}/analysis  (watch_session.py)
Phone (watch/phone)        ──POST /camera/trace─────▶ laptop server ──▶ data/camera/  (camera.py)
```

`GET /runs/{run}/analysis` returns every estimate (heart rate, HRV, ECG, motion) with its source,
tier and reliability (reference, good, experimental or unreliable), and the one it chose. For HRV
the priority is ECG, then SDK 25 Hz PPG, cleaned Samsung IBIs, 100 Hz PPG, and `hr_raw` last.

## Repository layout

| Path | Contents |
|---|---|
| `pulse_exploration/server.py` | FastAPI app: run upload, run list, health check |
| `pulse_exploration/watch_session.py`, `watch_api.py`, `watch_raw.py` | watch run analysis and `hr_raw` decoding |
| `pulse_exploration/camera.py`, `camera_api.py` | camera PPG analysis and upload endpoints |
| `pulse_exploration/pages_api.py`, `static/` | phone web pages |
| `pulse_exploration/hrv/` | beat-timing methods (`cand_*`), benchmark harness (`bench.py`), AF screen |
| `pulse_exploration/galaxyppg.py` | GalaxyPPG loader, ECG reference beats, watch-to-strap clock alignment |
| `pulse_exploration/ppg_age.py`, `net1d.py` | exploratory PPG age with AI-PPG Age: beat averaging, gates, model |
| `benchmarks/` | `hrv25`, `morph25`, `af25`, `afdb`, `watch_ecg` (one `run.py` each) |
| `watch/probe/`, `watch/phone/` | Wear OS recorder app and phone camera app (Kotlin, Gradle) |
| `docs/`, `reports/`, `research_notes/` | findings, research report and its notes |
| `plans/handoffs/` | session handoff with the full history of decisions |
| `tests/` | pytest suite, with synthetic watch and camera signals |
| `scripts/make_cert.py` | self-signed certificate for the HTTPS `/capture` page |

## Running it

Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest -q
uv run uvicorn pulse_exploration.server:app --host 0.0.0.0 --port 8765
```

The server has no authentication, so run it only on a trusted LAN. The `/capture` page needs HTTPS
for camera access: `uv run python scripts/make_cert.py` writes a certificate to `data/certs/` and
prints the command that serves the same app on port 8766.

The Android apps need the Android SDK and JDK 21:

```bash
cd watch && ./gradlew --no-daemon -q :probe:assembleDebug :phone:assembleDebug
```

The Samsung Health Sensor SDK is licensed separately and is never committed. Download it from
developer.samsung.com/health/sensor and put `samsung-health-sensor-api-*.aar` in `watch/probe/libs/`.
Without it (or with `-PsamsungSdk=false`) the probe builds a stand-in and records tier `25hz` only.
Installing and driving the apps over adb is described in [watch/phone/README.md](watch/phone/README.md)
and in the handoff in `plans/handoffs/`.

### Reproducing the benchmarks

Download GalaxyPPG from Zenodo (doi:10.5281/zenodo.14635823) and unpack it so that
`data/external/galaxyppg/unpacked/Dataset` exists, or set `GALAXYPPG_DIR`. The MIT-BIH benchmark
downloads PhysioNet annotation files into `data/external/physionet/` itself.

```bash
uv run python benchmarks/hrv25/run.py --workers 20   # 25 Hz HRV vs ECG
uv run python benchmarks/morph25/run.py              # pulse shape at 25 Hz
uv run python benchmarks/af25/run.py                 # AF screen false alerts on GalaxyPPG
uv run python benchmarks/afdb/run.py                 # AF screen on MIT-BIH
uv run python benchmarks/watch_ecg/run.py            # own watch: 25 Hz beats vs the watch ECG
```

Results go to `data/benchmarks/<name>/`.

### PPG age

`uv sync` installs CPU-only torch, which is enough for the 0.4 M-parameter model. The run analysis
never downloads the weights, so fetch them once:

```bash
uv run python -m pulse_exploration.ppg_age --download
uv run python -m pulse_exploration.ppg_age data/runs/<run>   # or a data/camera/<id>-finger-trace directory
```

## Data and licences

- `data/` is gitignored: watch runs, camera traces, downloaded datasets, caches and benchmark output.
- GalaxyPPG (Park, Zheng & Lee, *Scientific Data* 2025) is CC-BY-4.0. The MIT-BIH databases are under
  the Open Data Commons Attribution License v1.0.
- Face-video HR uses ME-rPPG through `open-rppg`. FacePhys is excluded from the server because of its
  licence.

## Status and next steps

- **Phase 0 checks still open:**
  - a full night of logging on battery;
  - a same-wrist 25 Hz vs 100 Hz recording, to settle pulse shape on the real device;
  - more watch-ECG recordings before `hr_raw` HRV is promoted from "experimental";
  - a Samsung Health data export, whose overnight beat-interval history is kept only briefly.
- **Phone app:** not yet run on a real phone.
- **Next phases, per the report's build plan:**
  - a nightly pipeline (resting heart rate, HRV percentile, respiratory rate, sleep, personal
    baselines);
  - the AF screen on our own nights, with ECG confirmation;
  - 30-second SDK spot checks for pulse-shape and vascular trends;
  - PPG age from existing models on repeated still spot checks, then embedding-based aging trends,
    both labelled as exploratory.
