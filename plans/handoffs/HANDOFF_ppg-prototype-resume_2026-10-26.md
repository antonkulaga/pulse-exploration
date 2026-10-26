# PPG/ECG prototype on Galaxy Watch + phone: two tiers built, 25 Hz HRV benchmarked, device checks pending

**Date:** 2026-10-26
**Status:** IN PROGRESS. Software is built and committed; the decisive on-device checks wait for the user's watch and phone.
**Bead(s):** none (no beads database in this repo)
**Epic:** PPG/ECG digital-health prototype (`~/sources/pulse-exploration`)
**Chain:** `standalone-4784afb3` seq `1`
**Parent:** `none — first in chain`
**Prior chain:** none — first in chain

Mining pass: Chunked (one very long session, 2026-10-21 to 10-24, hundreds of tool calls and eight
workflows). Part of the 2026-10-22 to 10-24 work (the finished benchmark, Phase 0 analysis, AF screen,
research report, CGM plan) ran in a stretch whose transcript was not in context when this was
written. Those parts come from the committed docs and commits, and are marked **[from repo docs]**.

---

## Reference Documents

- `docs/model-feasibility.md`: which open models fit the watch and phone, licences, gating, the
  access to-do list. Written 2026-10-22 from two research workflows plus the first watch probe.
- `docs/hrv-25hz-benchmark.md`: GalaxyPPG benchmark; 25 Hz HRV works at rest. **[from repo docs]**
- `docs/phase0-offline-findings.md`: `hr_raw` channel map, hidden lost samples, pulse shape at
  25 Hz. **[from repo docs]**
- `docs/af-screen.md`: irregular-rhythm screen, checked on GalaxyPPG and MIT-BIH. **[from repo docs]**
- `docs/cgm-track-plan.md`: plan for an N-of-1 CGM-paired study with Livia; nothing built.
  **[from repo docs]**
- `reports/PPG health and aging measures.md`: evidence-graded research report with the **build plan
  (phases 0 to 6)**. This is the roadmap; start there for "what next". **[from repo docs]**
- `research_notes/PPG health and aging measures/*.md`: five research notes behind that report
  (aging_vascular, camera_rppg, cardiometabolic, open_models_datasets, sleep_resp_autonomic).
- `watch/phone/README.md`: phone app install, adb extras, best-effort camera settings.
- Memory: `~/.claude/projects/-home-antonkulaga/memory/digital-health-ppg-prototype.md` and
  `python-uv-dependencies.md` (verified device facts; use uv).
- `~/sources/glucowatch/AGENTS.md`: the user's other Kotlin watch/phone project; same Gradle/AGP
  setup was copied from it.
- `README.md` in the repo is **empty**, so don't look for an overview there.

## The Goal

Build a **prototype** (not a product: the user said to ignore regulation, CE, FDA and GDPR) of a
digital-health app that measures cardiovascular signals from devices ordinary people already own:
- Samsung Galaxy Watch (the user's is a Watch6 Classic 47 mm LTE, SM-R965F);
- Galaxy S Ultra or latest Honor flagship phones.

No expensive extra equipment. A laptop server analyses what the devices record.

The guiding question has moved from "can we reuse the Manhph2211/PulseLM models" (answer: mostly
not) to "what can we measure **zero-shot**, without labelled data, from the watch and phone".
Every feature is labelled by how much the end user must do:
- **tier "25hz":** plain Android sensors, install only;
- **tier "all":** Samsung Health Sensor SDK, which needs Health Platform developer mode or Samsung
  partner approval.

The code uses the SDK when the watch allows it and falls back to what it can otherwise. The
longer-term product idea in the research report: a **personal-baseline engine** built on nightly
beat timing (resting HR, HRV percentile, respiratory rate, AF screen, illness deviations), with
30-second SDK spot checks as the route to vascular trends.

## Where We Are

### Watch: what is measured and verified on the user's watch
- **Watch app `watch/probe`** (`io.github.antonkulaga.pulseprobe`, Wear OS, Kotlin, plain Views):
  - All recording goes through `RecorderService.kt`, a foreground service of type **health** with a
    wake lock.
  - Commands, all sent through `MainActivity`: `caps`, `session`, `spot`, `probe`, `record`, `wifi`.
  - It holds Wi-Fi on purpose (`Network.kt`/`WifiHold`).
  - It uploads each run to the laptop with `PUT /runs/{run}/{file}`.
  - It writes `capabilities.json` into every run.
- **`Capabilities.kt` + `CapabilityProbe.kt`** decide the tier at startup, then map each feature to
  its preferred sources:
  1. Look up the Android sensors.
  2. Connect to the SDK, with a 10 s timeout.
  3. Run a 3 s HEART_RATE_CONTINUOUS policy test: `SDK_POLICY_ERROR` means denied, data or no error
     means allowed.

  Estimated timings, not yet measured on the watch: denied in about 1.5 s, allowed in about 3.5 s.
- **The SDK is optional at build time.**
  - `src/samsungSdk/.../SamsungSdk.kt` has the real code.
  - `src/noSamsungSdk/.../SamsungSdk.kt` is a stand-in.
  - `-PsamsungSdk=false` or a missing `.aar` builds the stand-in.
  - The `.aar` lives in `watch/probe/libs/` and is gitignored; it is licensed by Samsung and must
    never be committed.
- **The new capability-aware build (2026-10-22) has NOT been installed on the watch yet.** The
  installed APK is an older v2 (tier tags, SDK trackers, flush fix, Wi-Fi hold, upload). The watch
  dropped off adb before the capability version existed.
- **Tier "25hz" is verified on the real watch.**
  - `com.samsung.sensor.hr_raw` (type 69682) opens through plain SensorManager with only the
    `android.permission.health.READ_HEART_RATE` permission.
  - It delivers 16 int-packed channels at **24.9 Hz**. The values are float bit patterns, so they
    must be decoded with `floatToRawIntBits`.
  - Android HR runs at 1 Hz; accelerometer and gyroscope run at 99.6 Hz.
- **Tier "all" is verified on the real watch,** after the user enabled Health Platform dev mode:
  - `sdk_ppg`: green/IR/red at 24.95 Hz.
  - `sdk_hr`: 1 Hz with IBI lists.
  - `sdk_acc`: 24.9 Hz.
  - `sdk_ppg100`: 100 Hz on-demand, 99.6 Hz, 3283 samples in 30 s.
  - `sdk_skin`: gave 0 events in 30 s. It needs `READ_SKIN_TEMPERATURE`, which is now granted, and
    probably a longer run.
  - ECG and SpO2 have **not** been recorded yet.
- **Blocked to third-party apps,** because they need the signature-only `com.samsung.permission.SSENSOR`:
  - `hr_raw_fac` and `hr_raw_fac2` (100 Hz PPG);
  - `com.samsung.sensor.ecg` (TI AFE4500S);
  - `skin_temp` and `accelerometer_32g`.

  `registerListener` simply returns false for these.
- **`hr_raw` channel map** [partly from repo docs]: see the table in Evidence. The server uses v5
  (the AC channel, about green).

### Phone
- **Phone app `watch/phone`** (`io.github.antonkulaga.pulsephone`, CameraX 1.6.2):
  - **Fingertip mode:** rear camera, torch on.
  - **Face mode:** front camera with a guide box.
  - Settles for 2 s, then locks AE/AWB, confirmed from capture results.
  - Timestamps come from the sensor clock.
  - Posts per-frame RGB means (never video) to `POST /camera/trace`.
  - Offline traces are saved and can be retried.
  - Can be driven from adb.
- **Phone app status:**
  - **Tested on the emulator only** (AVD `glucowatch_phone`, read-only): 29.6–29.9 fps, no gaps,
    exposure and white balance lock confirmed.
  - **Never run on a real phone.**
  - Layout fixed: START is pinned, the result shows first and the view auto-scrolls to it.
  - Poor-quality heart rate shows as "(120 bpm?) NOT RELIABLE".

### Laptop server
- **FastAPI server `pulse_exploration/server.py`** runs on port 8765. **It is currently NOT
  running**; it stopped when the session process ended. Routes:
  - `/health`, `/runs`, `PUT /runs/{run}/{file}`;
  - `GET /runs/{run}/analysis` (`watch_api.py`);
  - `POST /camera/video`, `POST /camera/trace` (`camera_api.py`);
  - pages `/`, `/upload`, `/capture` (`pages_api.py` + `static/`). These are a web fallback; the
    user prefers the native app.
  - No authentication: LAN only.
- **`watch_session.py`** (ANALYSIS_VERSION `watch-session-0.3`) analyses a watch run.
  - It reports every estimate with source, tier and reliability (reference / good / experimental /
    unreliable), plus the chosen one.
  - **HRV priority:** ECG > `sdk_ppg` > Samsung IBI (cleaned) > 100 Hz PPG > `hr_raw`.
  - 25 Hz HRV uses `foot_template` plus a per-axis accelerometer gate (1.0 m/s² over 2 s).
  - `HRV25_BEST = {"sdk_ppg": "good", "hr_raw": "experimental"}`: `hr_raw` is capped until it is
    checked against the watch ECG. `MOTION_RMS = 0.5`. [0.3 is partly from repo docs]
- **`camera.py`** does the camera-PPG analysis:
  - **Classical:** inverted red/green for the fingertip, POS/CHROM for the face.
  - **Pretrained:** ME-rPPG via open-rppg (`DEEP_MODELS = ("ME-chunk.rlap",)`). FacePhys is
    excluded on purpose (`SERVER_FORBIDDEN_MODELS`).
  - **Honesty checks:** no fingertip forces quality to poor; a face amplitude above
    `FACE_MAX_REL_AMPLITUDE = 0.012` forces poor; a deep-model HR that disagrees with POS/CHROM by
    more than 5 bpm gets `deep_unconfirmed`; key-frame-rate lock-on is flagged.

### Benchmarks and research
- **The 25 Hz HRV benchmark on GalaxyPPG is done** [from repo docs]. `foot_template` and
  `motion_denoise` give about **3 ms median RMSSD error** against chest-strap ECG on held-out
  participants. They cover 75–79% of 60 s windows. Nothing covers walking or running. Details in
  Evidence.
- **Other finished pieces** [from repo docs]:
  - AF screen `pulse_exploration/hrv/af_screen.py`, tested on GalaxyPPG and MIT-BIH;
  - pulse-shape-at-25-Hz test `benchmarks/morph25`;
  - research report;
  - CGM plan.
- **Tests:** 134 passed after the capability work (reviewer's count, 2026-10-22). Later commits added
  `test_af_screen.py` and candidate tests; the full suite has not been re-run since. Run
  `uv run pytest -q` first.

### Repo and environment
- **Git:** branch `master`, **no remote**, working tree clean. Last commit `7915bdb` (2026-10-24).
  Seven commits in all (table in Evidence). The global CLAUDE.md says "commit and push to main
  without asking", but there is nothing to push to.
- **Devices:** none connected over adb now. The watch and phone both need Wireless debugging
  re-enabled.
- **Data on disk** (gitignored `data/`, about 2.5 GB):
  - `data/runs/` holds 6 real watch runs from 2026-10-22 plus `hrv_bench/`.
  - `data/external/galaxyppg/` (1.8 GB unpacked) and `data/external/physionet/` (annotations only).
  - `data/cache/galaxyppg/` (141 MB parquet), `data/benchmarks/`, `data/camera/` (emulator traces),
    `data/screens/` (emulator screenshots), `data/models/` (YuNet ONNX), `data/certs/`.

## What We Tried (Chronological)

1. **Model review (2026-10-21).** Research workflow 1 had 19 agents, workflow 2 six.
   - Covered: github.com/manhph2211/PulseLM, arXiv 2603.03331, and HF Manhph2211 (PulseLM, CP-PPG,
     D-BETA, Q-HEART, ECG-Scan, Cheap2Chic), plus PaPaGei and Pulse-PPG.
   - Result: none works zero-shot on watch or phone data.
   - PulseLM is Qwen2.5-7B plus the Pulse-PPG encoder, 15.3 GB bf16, gated-manual. It gives
     categorical answers only. By the paper's own v2 Table 11 it does not beat the text-only "Blind"
     baseline on AF (85.4 vs 85.3) or RR (32.7 vs 33.1). There is no label leak.
   - Cheap2Chic is an earphone audio model with an empty repo.
   - D-BETA, Q-HEART and ECG-Scan need 12-lead input or synthetic printouts.
   - Written up in `docs/model-feasibility.md`.
2. **Zero-shot path search.** Better-suited open models turned up:
   - ECGFounder (MIT, single-lead, 150 labels);
   - Pulse-PPG (MIT, pretrained on Wear OS watch PPG);
   - AnyPPG (no licence);
   - ME-rPPG (MIT);
   - PPG-VascularAge / AI-PPG Age (MIT).

   An adversarial check found two important limits. The SDK's on-demand trackers run at most 30 s,
   one at a time, and continuous trackers may be invalid meanwhile. And the stock-camera fingertip
   claims were inferred rather than verified.
3. **Wireless adb to the watch.**
   - Discovery: `stdbuf -oL avahi-browse -rp _adb-tls-connect._tcp` and `_adb-tls-pairing._tcp`.
     `adb mdns services` found nothing.
   - The first `adb pair` failed with "protocol fault" because the watch was dozing; a retry
     worked.
   - The watch's adb port changes every time Wireless debugging is toggled: 33125, 39085, then
     42091.
   - **Wireless debugging keeps switching itself off** even while Wi-Fi stays up. Holding Wi-Fi did
     not prevent this.
4. **Probe app v1, a plain Activity.** It failed four ways, each fixed in turn:
   1. A crash: `SENSOR_DELAY_FASTEST` needs the `HIGH_SAMPLING_RATE_SENSORS` permission.
   2. Doze stalls Handler timers and non-wake-up sensors. A wake lock alone was **DISABLED** in
      ambient mode.
   3. → Moved recording into a **foreground service of type health**, which fixed both.
   4. Samsung's `MARsmini_FreecessController` froze the app when it was launched with the screen
      off. → Send `input keyevent KEYCODE_WAKEUP` first and launch with
      `am start -S -f 0x10008000` (clear task); otherwise the intent is delivered to a dead task.
   5. A POST_NOTIFICATIONS dialog appeared and the user denied it (USER_SET). → Notifications are
      now optional and not required for a run.
5. **The `hr_raw` discovery.**
   - `dumpsys sensorservice` showed `Samsung HR Raw Sensor` guarded only by READ_HEART_RATE. Samsung's
     2024 blog claims third-party apps get null from such sensors, but on this firmware it works.
   - The values arrive as float bit patterns (`1.4E-45` = int 1, `2.9E-39` = 2097152 = 2^21).
   - The first 60 s on-wrist recording gave HR 56.6 bpm (peaks) / 55.7 (FFT) against the watch's
     57 median.
   - RMSSD from naive peaks was 273 ms, which is non-physiological.
6. **The SDK_POLICY_ERROR saga.**
   - Every SDK tracker returned `SDK_POLICY_ERROR`. The user said dev mode was on.
   - Logcat `SHS#TrackerRequestHandler: isDeveloperMode false` proved it was not.
   - **My instructions had named the wrong app.** The pulled APK label is "Health Platform",
     package `com.samsung.android.service.health`, and it shows "[Dev mode]" once enabled. It is not
     "Health Sensor Service".
   - After the user enabled it, the log said `isDeveloperMode true`, but still 0 events. This was
     **my bug**: `flush()` was immediately followed by `unsetEventListener()`, so the flushed batch
     arrived with no listener. The fix: flush in `stop()`, wait `FLUSH_WAIT_MS = 3000`, then unset
     in `close()`.
   - Skin temperature's `PERMISSION_ERROR` meant `android.permission.health.READ_SKIN_TEMPERATURE`
     was missing (Android 16 granular permission). Added together with `READ_OXYGEN_SATURATION`.
7. **Cross-tier alignment.**
   - The first correlation between `hr_raw` and `sdk_ppg` came out at about 0. Cause: the SDK's
     `wall_ms` is batch *arrival* time, about 8 s late.
   - With `timestamp_ms`, v1/v4/v5 against SDK green/IR/red reach r = 0.93–0.97 at 0–80 ms lag.
   - SDK HR and Android HR are the same algorithm and match value for value.
8. **The 100 Hz on-demand recording.**
   - Green, IR and red were all valid except one red sample.
   - HR 61.2 bpm.
   - RMSSD 271.8 ms and SDNN 197.5 ms, which is nonsense, caused by the IR start-up ramp and
     artifacts.
9. **Camera backend** (workflow, 5 agents).
   - Built `camera.py`, `camera_api.py`, the web pages and 88 tests. YuNet was chosen over MediaPipe
     because MediaPipe's `opencv-contrib` clashes with open-rppg's `opencv-python`. open-rppg needs
     `setuptools<82` (it imports `pkg_resources`).
   - **The FacePhys weights' Privacy Addendum (2026-08-03) forbids server-side inference.** I read
     the FacePhys-Release LICENSE text myself, so FacePhys was removed from the server.
   - That left a single deep model, so "models disagree" could never fire. → Added
     `deep_unconfirmed`, which compares against POS/CHROM.
10. **Phone app** (workflow).
    - Built and reviewed; 5 bugs fixed, including the torch going off after a camera interruption
      and the AE-lock frame skip.
    - Emulator debugging found more:
      - the result was hidden below the settings (layout fixed);
      - "quality GOOD 120 bpm" with no fingertip (→ `finger_missing` forces poor);
      - face "180 bpm GOOD" from a blinking scene at 1.57% amplitude. The 2% threshold was too
        loose and was tightened to 1.2%.
    - The user's GlucoWatch app inside the AVD popped a Bluetooth-permission dialog. It was
      force-stopped in that read-only session only.
11. **Beat-by-beat 25 Hz against Samsung IBI on the 30 s two-tier run: inconclusive.**
    - Mean |diff| 112 ms, r = 0.02.
    - Order-based alignment of IBI lists fails because Samsung drops intervals between batches and
      gives no per-beat timestamps. Samsung's own SDNN of 109 ms in 30 s looked artifact-laden.
12. **SDK-or-fallback** (workflow, 3 agents). Built Capabilities, the `session`/`spot`/`caps`
    commands, `watch_session.py` and `/runs/{run}/analysis`. The reviewer fixed:
    - sources with 0 events being listed;
    - Samsung IBI dropped-beat detection;
    - "good" labels given without any motion check.

    134 tests passed.
13. **GalaxyPPG benchmark** (workflow).
    - Downloaded from Zenodo 14635823: 0.48 GB, MD5 `d9a48bfcd07928fb22b96dc40cf2e7d4` OK.
    - The harness and the naive and classical candidates finished.
    - **The foot_template, motion_denoise, score and verify agents failed on 2026-10-22 when the
      weekly usage limit was hit.**
    - The work was completed later [from repo docs]: their code and tests were complete, and their
      defaults were scored.
14. **After the benchmark** [from repo docs]. Server 0.3:
    - foot_template for 25 Hz HRV;
    - per-axis accelerometer gate;
    - `MOTION_RMS` raised from 0.2 to 0.5, because 0.2 called a still run 36% motion;
    - Samsung IBI cleaning, plus a 6 s shift back for its motion check;
    - an `hrv_sources_disagree` warning.

    Other work in that stretch:
    - **Rejected:** gating the lost-sample rule on how calm the neighbouring intervals are. It fixed
      the high-HRV bias but worsened everyone else's error (3.5 → 5.4 ms on P02–P06).
    - **The `hr_raw` timestamp finding:** `hr_raw` respaces its stamps, which hides about 2 lost
      samples per 30 s.
    - **The AF screen:** the HRV cleaning hides AF, so it uses `AF_SETTINGS`, and a trimmed RMSSD
      cuts false positives.
    - **The research report** and the **CGM plan**.

## Key Decisions

- **Two tiers, named "25hz" and "all".** The user asked for this explicitly ("so I would know what
  will need extra steps from users"), and every output carries its tier.
  - The phone camera app also needs no extra user steps.
  - **Rejected:** making the SDK mandatory. Developer mode on every user's watch is "extra
    headache", and partner approval is needed for distribution.
- **Use the SDK when allowed, otherwise fall back** (user's request): detect at run time, and also
  make the `.aar` optional at build time.
- **Native Kotlin phone app over web pages.** The user asked "why do we need some webpage on phone
  if the app can do stuff". The web pages stay only as a fallback.
- **The server does the analysis; the phone and watch only capture.** The user said "we can make an
  API on a server that phone can deal with, the key part is to make it work".
- **No FacePhys on the server:** its licence forbids it and consent cannot waive that. It may come
  back on the phone as a 1.15 MB tflite.
- **Honesty over numbers.** Poor quality is shown as "(120 bpm?) NOT RELIABLE". 25 Hz HRV is never
  better than "experimental" for `hr_raw` until it passes the ECG check.
- **The fingertip video is never uploaded**, only per-frame RGB means plus metadata.
- **ECG cannot be unlocked by code** (signature permission). HRV *can*, through better 25 Hz signal
  processing, and that turned out true at rest. **Rejected:** bypassing Samsung's permissions or
  rooting the watch.
- **Rejected:** fine-tuning PulseLM, using CP-PPG before vascular features (it pushes waveforms to a
  young-adult template), and Q-HEART or D-BETA on zero-padded single-lead ECG for accuracy claims.
- **Research-report "do not build" list** [from repo docs]: calibration-free BP in mmHg, glucose
  values, lipids, face-video BP/SpO2/age, carotenoid/AGE indices, and any absolute vascular age.
  Per a later project decision, PPG age may be shown with its error band and a warning.
- **Aging clocks are judged by outcome evidence** (gap → disease or death), not by MAE. This is the
  user's explicit view (from memory).

## Evidence & Data

### Watch hardware (SM-R965F, from `dumpsys sensorservice`, 2026-10-22)
| Sensor | Type / string | Permission | Rate | Open to third-party apps? |
|---|---|---|---|---|
| Samsung HR Raw | 69682 `com.samsung.sensor.hr_raw` | health.READ_HEART_RATE | 25 Hz, FIFO 600 | **Yes** (24.9 Hz measured) |
| HR Raw Fac / Fac2 | 69683 / 69684 | SSENSOR | 100 Hz | No |
| AFE4500S ECG | 69669 `com.samsung.sensor.ecg` | SSENSOR | 100 Hz events (SDK gives 500 Hz) | No |
| AFE4500S BIA | 69680 | SSENSOR | 100 Hz | No |
| Skin Temp | 69686 | SSENSOR | ≤1 Hz | No (SDK only) |
| LSM6DSO Accelerometer 32G | 69657 | SSENSOR | ≤100 Hz | No |
| HR Batch / HR None Wakeup | 21 `android.sensor.heart_rate` | READ_HEART_RATE | 1 Hz | Yes |
| Accelerometer / Gyroscope (AOSP/STM) | standard | none | 99.6 Hz measured | Yes |

Other build facts:
- Android 16, API 36, One UI Watch 8 (`ro.build.version.oneui` 80000).
- Build BP2A.250325.020.R965FXXS2CZF5, security patch 2026-07-05, CSC ROM.
- Health Platform 1.7.00.05 (`com.samsung.android.service.health`, label "Health Platform" / WearSTS).
- SDK reports version 1.4.01.00.
- Samsung Health on the watch (`com.samsung.android.wear.shealth`) has **no ALLOW_BACKUP**, so adb
  cannot pull its history.
- 2.7 GB free on the watch.

SDK supported types on this watch: PPG_GREEN, ACCELEROMETER(_CONTINUOUS), ECG(_ON_DEMAND), PPG_IR,
PPG_RED, SPO2(_ON_DEMAND), HEART_RATE(_CONTINUOUS), BIA(_ON_DEMAND), SWEAT_LOSS,
SKIN_TEMPERATURE(_ON_DEMAND/_CONTINUOUS), PPG_ON_DEMAND, PPG_CONTINUOUS. EDA is not supported
(Watch8 and later).

### Real watch recordings (`data/runs/`, all 2026-10-22, on the user's wrist)
| Run | Contents | Key result |
|---|---|---|
| `20261022-013027-probe` | probe, SDK dev mode off | hr_raw 24.88 Hz, HR 59; all SDK rows `SDK_POLICY_ERROR`; skin `PERMISSION_ERROR`; EDA unsupported |
| `20261022-013105-record` | 60 s hr_raw, hr, acc, gyro (25hz) | HR 56.6 peaks / 55.7 FFT vs watch 57 median (53–63); about half the minute spoiled by motion; `ppg.png` plot |
| `20261022-013245-record` | 5 s sdk_hr, sdk_ppg (diagnostic) | used only to read the `isDeveloperMode false` logs |
| `20261022-015506-probe` | probe, dev mode on | `isDeveloperMode true`; 0 SDK events (flush bug) |
| `20261022-015638-record` | 30 s hr_raw, hr, acc + sdk_ppg, sdk_hr, sdk_acc, sdk_skin | both tiers; HR 57→67→60 identical in sdk_hr and hr; sdk green SNR 54 at 58.6 bpm |
| `20261022-015808-record` | 30 s sdk_ppg100 + acc | 3283 samples at 99.6 Hz; HR 61.2; HRV nonsense |

Analysis endpoint results (server 0.2/0.3 at the time; may change on recompute):
| Run | Tier | HR chosen | HRV chosen |
|---|---|---|---|
| 013105 | 25hz | 57.0 (hr, good) | hr_raw RMSSD 31.1 / SDNN 72.1, 16 beats, 15.3 s: unreliable (30% still) |
| 015638 | all | 61.0 (sdk_hr, good); also hr 60.5, sdk_ppg 58.9, hr_raw 59.2 | sdk_hr RMSSD 74.7 / SDNN 103.3: experimental; sdk_ppg 48.4, hr_raw 49.8 unreliable |
| 015808 | all | 60.4 (sdk_ppg100, experimental) | none (motion gate rejected all, at the old 0.2 threshold) |

Later analysis of 015638 [from repo docs]:
- sdk_ppg 59 ms and hr_raw 63 ms agree.
- Cleaned Samsung IBI gives 36 ms against 69 ms raw. The lost-sample rule trims genuine intervals at
  the user's high resting RMSSD of about 60 ms.

### `hr_raw` channel map [phase0 doc + this session]
| Channel | Behaviour | Best match | Reading |
|---|---|---|---|
| v0 | 1 off wrist, 2 on wrist | — | status |
| v1, v4 | about 2.7×10⁶, pulsatile | IR r = 0.97, green 0.95 | raw light level, closer to IR/red |
| v5 = v7 | 2^21 offset, largest swing | **green r = 0.98** | AC pulse channel; used by the server |
| v9, v11, v13 | pulsatile | IR/red r = 0.96–0.97 | probably IR/red |
| v6, v8, v10, v12, v14 | about 2^21, no pulse | — | ambient or idle |
| v2, v3, v15 | 0 | — | unused |

- `hr_raw` stamps sit at 40.16 ms median, evenly respaced, which hides about 2 lost samples per
  30 s.
- The SDK stream sits on a 40.00 ms grid and shows its losses as gaps (two of about 75 ms in 30 s).
- hr_raw against SDK green, beat by beat: a constant about 60 ms offset plus a sawtooth (about
  4 ms per beat ramp, ±40 ms jumps). The interval-difference SD is 24 ms.
- Samsung IBIs on this watch lag beats by 5.2–7.0 s.

### 25 Hz HRV benchmark on GalaxyPPG, held-out P07–P24, 60 s windows [from repo docs]
| Method | Covered | RMSSD median error | Bias (95% limits) | Within ±10 ms | Yield |
|---|---|---|---|---|---|
| naive | 99% | 103 ms | +182 (−233…+597) | 3% | 3% |
| classical | 63% | 5.2 ms | +5.5 (−5…+16) | 91% | 58% |
| **foot_template** | 79% | **3.1 ms** | +2.5 (−6…+11) | 95% | 75% |
| motion_denoise | 75% | 3.0 ms | +1.9 (−8…+12) | 95% | 71% |
| samsung_ibi (SDK, raw) | 77% | 10.0 ms | +32 (−68…+133) | 50% | 38% |
| samsung_ibi_clean (SDK) | 72% | 4.4 ms | +1.5 (−11…+14) | 91% | 65% |

- **300 s windows** (34, seated rest): foot_template covered 100%, 2.4 ms error, 100% within
  ±10 ms.
- **By activity, foot_template:**
  - seated rest 94% covered, 2.7 ms;
  - cognitive stress 84%, 3.3 ms;
  - standing 92%, 5.5 ms;
  - typing 49%, 5.0 ms;
  - walking, jogging and running declined by every non-naive method.
- **GalaxyPPG data facts:**
  - 24 people aged 20–29.
  - P01 has no watch data; P11's ECG is unusable; P07 and P08 have no Samsung HR file.
  - The watch stream silently loses about 1 sample every 10–11 s (40.15 ms mean spacing). Samsung
    IBIs trail beats by about 7.4 s (6.3–8.5 s).
- **Fairness:** tuned on P02–P06, frozen before the held-out run. The review found no leakage.
- **Reproduce:** `uv run python benchmarks/hrv25/run.py --workers 20`. Results go to
  `data/benchmarks/hrv25/`.

### Naive vs Samsung IBI, from the harness agent (this session, before the limit hit)
| Source, window | Coverage | Ref RMSSD median | RMSSD median abs error | Bias (LoA) |
|---|---|---|---|---|
| naive 60 s | 0.99 | 24.8 | 112 ms | +186 (−226…+599) |
| naive 60 s seated rest | 1.00 | 27.7 | 28 ms | +147 |
| Samsung IBI 60 s | 0.76 | 25.8 | 10.5 ms | +34 (−68…+136) |
| Samsung IBI 300 s | 0.98 | 31.4 | 15.5 ms | +28 (−57…+113) |

Clock-alignment offsets per participant ranged from −5506 ms (P15) to +1556 ms (P23), with drift −16
to −54 ppm, fitted from an R-peak-triggered PPG-derivative average. Of 1,044 60 s windows, 898 are
usable.

### AF screen [from repo docs]
- **Rule:** 64-beat segments; trimmed RMSSD/meanNN ≥ 0.13 (8 largest successive differences
  dropped) and normalised Shannon entropy ≥ 0.76; an alert needs 5 of 6 consecutive segments
  irregular. Based on Lee et al. 2013.
- **GalaxyPPG** (healthy, sinus rhythm): 0 alerts. Chest ECG gave 0.08% irregular segments; 25 Hz
  PPG gave 0%.

MIT-BIH AF + NSR, 5 ms jitter:
| Rule | Sens | Spec (AF patients' sinus) | Spec (healthy) | AF records alerted | False alerts/day healthy |
|---|---|---|---|---|---|
| Lee untrimmed 0.13 | 99.2% | 77.2% | 97.9% | 20/23 | 0.96 |
| Trimmed 0.13 (default) | 87.6% | 94.4% | 99.7% | 21/23 | 0.13 |
| Trimmed 0.09 (AF-burden) | 95.9% | 89.9% | 98.2% | 21/23 | 1.22 |

### Pulse shape at 25 Hz (`benchmarks/morph25`, GalaxyPPG E4 64 Hz → 25 Hz → rebuilt) [from repo docs]
| Index | Error SD, band-limited | Between-person spread | Ratio |
|---|---|---|---|
| Crest time | 0.4 ms | 44 ms | 0.01 |
| Width at half height | 0.9 ms | 36 ms | 0.03 |
| b/a | 0.001 | 0.068 | 0.02 |
| d/a | 0.005 | 0.042 | 0.12 |

Caveat: the E4 signal is already filtered below 10 Hz, so this tests reconstruction, not content the
E4 never had.

### Camera analysis, adversarial synthetic tests (this session)
- **Fingertip:** within ±1–3 bpm for HEVC CRF 28, 25/30/60 fps, 45 and 150 bpm, flicker,
  exposure drift and steps, jitter, variable frame rate and clipped red.
- **Fingertip failures are flagged poor:** CRF 35, motion bursts, weak pulse, no pulse.
- **Face from compressed video:** every harsh case locks onto about 60 bpm, the key-frame rate.
  These are flagged poor honestly, but the result isn't useful.
- **open-rppg on CPU** (jaxlib has no CUDA): FacePhys 72.0 bpm SQI 0.77, load 15–20 s; ME-chunk
  72.1 bpm SQI 0.83, inference 2–3 s per 30 s video.

### Phone app on the emulator (synthetic scene, no real pulse)
| Mode | fps | Server result after fixes |
|---|---|---|
| Fingertip | 29.2–29.9, 1 gap | "(120 bpm?) NOT RELIABLE", quality POOR, `finger_not_detected` |
| Face | 29.6–29.9, 0 gaps | "(180 bpm?) NOT RELIABLE", POOR, `implausible_face_amplitude` (1.6%) |

Screenshots are in `data/screens/` (`phone-app-emulator.png` is the three-panel one).

### Commits (all on `master`, no remote)
| Hash | Date | Summary |
|---|---|---|
| fe28829 | 2026-10-22 16:44 | Add PPG/ECG prototype and 25 Hz HRV benchmark on GalaxyPPG (77 files, 18,360 lines) |
| 8807ebb | 2026-10-22 17:12 | Use foot_template for 25 Hz HRV and clean Samsung's IBIs in the server |
| c00bc2f | 2026-10-24 03:11 | Add cardiometabolic PPG/ECG detection research notes |
| 66b19b9 | 2026-10-24 03:25 | Add research report on PPG health and aging measures |
| 09a584c | 2026-10-24 03:26 | Add research notes behind the PPG health and aging report |
| 55b0b3e | 2026-10-24 11:48 | Add AF screen, 25 Hz pulse-shape test and phase 0 findings |
| 7915bdb | 2026-10-24 12:40 | Check the AF screen on MIT-BIH AF and normal-rhythm records |

### Model verdicts (docs/model-feasibility.md, condensed)
| Model | Verdict for our devices | Access |
|---|---|---|
| PulseLM (Qwen2.5-7B + Pulse-PPG, 15.3 GB) | categorical QA only; only the HR category is credible; little value | gated-manual |
| CP-PPG (1.1 M U-Net, 128 Hz) | contact-pressure enhancer; erases vascular-aging morphology | gated-auto, CC BY-NC |
| D-BETA / Q-HEART | 12-lead 500 Hz; single lead out of distribution; Q-HEART plumbing only (needs Meta's Llama-3.2-1B gate) | manual / open + Llama gate |
| ECG-Scan | synthetic 12-lead printout embeddings only | gated-manual, CC BY-NC |
| Cheap2Chic | unrelated audio model, empty repo | — |
| ECGFounder | best zero-shot single-lead ECG (150 labels) | MIT, ungated |
| Pulse-PPG | best open encoder for watch PPG (Wear OS pretraining) | MIT, Zenodo |
| PaPaGei / AnyPPG | embeddings; weights unlicensed | Zenodo / repo |
| ME-rPPG | face HR on the server | MIT |
| FacePhys | best face model, **server use forbidden** | MIT + Privacy Addendum |

## Code Analysis

- `watch/probe/.../Sources.kt`:
  - `AndroidSensorSource` writes `event_ns,wall_ms,accuracy,v0..` and decodes ints for `hr_raw` via
    `floatToRawIntBits` (`Catalog.packedInts`).
  - `SamsungSource` writes `timestamp_ms,wall_ms,<columns>`. `stop()` only flushes; `close()`
    unsets the listener. IBI lists are joined with `|`.
  - `Catalog.samsung`: the SDK specs (sdk_ppg, sdk_hr, sdk_acc, sdk_skin, sdk_eda, sdk_ppg100,
    sdk_ecg, sdk_spo2).
- `RecorderService.kt` constants:
  - `PROBE_SECONDS = 8`, `SDK_TIMEOUT_MS = 10_000`, `FLUSH_WAIT_MS = 3_000`;
  - the wake lock is held for (seconds + 90) s;
  - `server` is persisted in SharedPreferences (`--es server URL` once).
- `RecorderService.PERMISSIONS`: READ_HEART_RATE (API 36) or BODY_SENSORS, ACTIVITY_RECOGNITION,
  READ_ADDITIONAL_HEALTH_DATA, READ_SKIN_TEMPERATURE, READ_OXYGEN_SATURATION.
- **Manifest:** FGS `health`, WAKE_LOCK, HIGH_SAMPLING_RATE_SENSORS, INTERNET,
  CHANGE_NETWORK_STATE, `usesCleartextTraffic=true`, and a `<queries>` entry for Health Platform.
- `capabilities.json` schema 1:
  - `{tier, sdk{installed,connected,policy,error,version,supported}, android{...}, features{ppg,hr,ibi,motion,skin_temp,ppg100,ecg,spo2: {available,tier,sources,reason}}}`
  - Sources by tier:

    | Feature | Tier "all" | Tier "25hz" |
    |---|---|---|
    | ppg | `[sdk_ppg, hr_raw]` | `[hr_raw]` |
    | hr | `[sdk_hr, hr]` | `[hr]` |
    | ibi | `[sdk_hr]` | unavailable |
    | motion | `[acc, gyro]` | `[acc, gyro]` |
- `pulse_exploration/watch_session.py`:
  - `analyze_run(run_dir)` returns `metrics.{hr,hrv,...}.{estimates[],chosen}` and writes
    `analysis.json`. `MOTION_RMS = 0.5`. `HRV25_BEST` caps hr_raw at experimental.
  - `/runs/{run}/analysis?refresh=1` recomputes.
- `pulse_exploration/hrv/`:
  - `bench.py`: the Estimator protocol, `estimate(ppg_t_ms, ppg, acc_t_ms, acc_xyz) -> {beats_ms, quality?, accept_windows?}`.
  - `cand_foot_template.py`: the production 25 Hz method.
  - `af_screen.py` with `AF_SETTINGS`: no interval artifact correction, otherwise HRV cleaning hides
    AF.
- `pulse_exploration/camera.py`:
  - `analyze_trace(t_s, r, g, b, mode, tier)` and `analyze_video(path, mode, ...)`;
  - constants `GRID_FS = 60`, `HR_BAND = (0.65, 3.5)`, `MIN_HRV_S = 60`, `MIN_HR_S = 20`,
    `FACE_MAX_REL_AMPLITUDE = 0.012`, `DEEP_MODELS = ("ME-chunk.rlap",)`;
  - `_deep_check` flags key-frame lock, model disagreement and `deep_unconfirmed`.
- `watch/phone`: `PpgCamera.kt` (CameraX + Camera2Interop), `FrameAnalyzer.kt` (RGBA ROI means,
  clipping), `CaptureLog.kt` (capture-result matching, AE/AWB lock wait), `TraceStore.kt`
  (save/upload/retry), `ResultFormatter.kt`. A poor-quality result gets the headline "(N bpm?)".

## Files Changed

### Source code (all new this session; see git for detail)
- `watch/probe/**`: Wear OS probe app with both tiers, capability detection, SDK source sets,
  Wi-Fi hold and upload.
- `watch/phone/**`: Kotlin CameraX phone app.
- `watch/settings.gradle.kts`, `watch/build.gradle.kts`, wrapper: Gradle 9.1.0, AGP 9.0.1, built-in
  Kotlin, `jvmToolchain(21)`.
- `pulse_exploration/server.py`, `watch_api.py`, `watch_session.py`, `watch_raw.py`,
  `camera.py`, `camera_api.py`, `pages_api.py`, `static/*`, `galaxyppg.py`, `hrv/*`.
- `scripts/make_cert.py`: self-signed certificate for the HTTPS `/capture` page.
- `benchmarks/hrv25|morph25|af25|afdb/run.py`.

### Tests
- `tests/test_camera.py`, `test_adversarial.py`, `test_pages.py`: camera and pages, synthetic
  videos only.
- `tests/test_watch_session.py`, `test_watch_adversarial.py`, `watch_synth.py`: run analysis,
  Kotlin-format cross-check.
- `tests/test_hrv_bench.py`, `test_cand_*.py`, `test_af_screen.py`: benchmark harness, candidates
  and AF screen.

### Docs and reports
- `docs/*.md` (5), `reports/PPG health and aging measures.md`, `research_notes/.../*.md` (5).

### Data (gitignored)
- `data/runs/*`, `data/camera/*`, `data/screens/*`, `data/external/galaxyppg`,
  `data/external/physionet`, `data/cache/galaxyppg`, `data/benchmarks/*`, `data/models/` (YuNet
  ONNX, MIT, sha256 `8f2383e4…`).

### Outside the repo
- Memory: `digital-health-ppg-prototype.md` and `python-uv-dependencies.md` added, plus MEMORY.md
  lines.
- Samsung SDK zip kept at `~/soft/samsung-health-sensor-sdk-v1.4.1.zip`.
- Emulator side effect: running the `glucowatch_phone` AVD read-only rewrote launch metadata and
  timestamps in `~/.android/avd/glucowatch_phone.avd/`; no app or data persisted.

## User Feedback & Preferences (REQUIRED — never omit)

- "if you want to handle dependencies use uv"; "maybe we can even create
  /home/antonkulaga/sources/pulse-exploration uv project" → all Python work goes there through uv.
- "ignore regulations, we are making a prototype, not a product".
- "we want to integrate stuff into digital health applications with just simple stuff that people
  have: samsung galaxy ultra or latest honor level phones, samsung watch, we do not want expensive
  equipment".
- "we can make an API on a server that phone can deal with, the key part is to make it work".
- "let's first see what we can get zero-shot as I just have my watch and phone, not thousands of
  datapoints".
- "if you need to deal with watch and phone we previously use kotlin and things worked ok (see
  ~/sources/glucowatch)" and "maybe we first have to probe dealing with it".
- "But we have the kotlin app, why do we need some webpage on phone if the app can do stuff" →
  native app first.
- "also 25hz you can start working we have to actually prioritize stuff that can run without extra
  headache from user".
- "use all of them, just try to have two feature tiers: 25hz tier and all tier so I would know what
  will need extra steps from users".
- "can we make a code in a way that if we have samsung sdk it uses it, if not it unlocks what it
  can".
- "also we can download GalaxyPPG" (explicit permission given).
- "you can debug phone app via emulator and produce screenshots or you can connect to my phone via
  wireless debugging".
- **Frustration** when my instructions named the wrong dev-mode app: "But i do have it enabled,
  wtf". Lesson: verify device-specific UI names (pull the APK label) before telling the user where
  to tap.
- Questions they wanted answered plainly:
  - "should it work at all?" (fingertip);
  - "how does it compare to what I get with wearos watch?";
  - "why watch requires SDK for HRV and ECG?";
  - "if we provide enough code can we unlock HRV without samsung SDK?"
- The user pasted a Hugging Face token in chat on 2026-10-21. I refused to use it and advised
  revoking it. **Never ask for tokens**; the user runs `hf auth login` themselves.
- From memory (later stretch): aging clocks are judged by outcome evidence, not MAE. PPG age may be
  shown with its error band and a warning.
- Global `~/.claude/CLAUDE.md`: commit (and push to main) without asking, only the task's files.
  There is no remote here.

## Where We're Going

Ordered by the research report's Phase 0 and by what the user has been asked for. Everything in 1–4
needs the user's devices.

1. **Reconnect the watch and install the capability-aware probe build.**
   - Ask the user to re-enable Wireless debugging (no re-pairing needed). Find the port with
     `avahi-browse`.
   - Ask again whether to run `adb tcpip 5555`, which gives a fixed port until reboot. The user
     never answered.
   - Install `watch/probe/build/outputs/apk/debug/probe-debug.apk`; rebuild first with the SDK
     `.aar` present.
   - Run `cmd caps` and check `capabilities.json` against the denied and allowed cases.
2. **Phase 0, decisive check: a 30 s watch ECG with `hr_raw` running at the same time.**
   - First check that ECG_ON_DEMAND and SensorManager `hr_raw` can run concurrently (separate
     paths; unverified).
   - The user rests their forearm and holds a fingertip of the other hand on the top key, then says
     "ready".
   - Score `hr_raw` beat timing against the ECG R-peaks. If it is within benchmark limits, set
     `HRV25_BEST["hr_raw"] = "good"`.
3. **Phase 0, the rest:**
   - an all-night `hr_raw` session (battery, Wear OS background limits);
   - a still same-wrist 25 vs 100 Hz recording;
   - a Samsung Health "Download personal data" export, soon, because the `sleep_raw_data`
     `rri_info` history probably covers only about a week. Parse it with
     `Livia-Zaharia/samsung_health_extractor`, keeping `.hrv`, `.stress`, `.oxygen_saturation.raw`,
     `.sleep_stage`, `.shm_device` and `.calibration_blood_pressure`.
4. **Phone on a real device.**
   - Pair over Wireless debugging (Samsung: About phone → Software information → Build number ×7;
     Honor: About phone → Build number ×7, then System & updates → Developer options). The user
     sends only the 6-digit code.
   - Install `watch/phone/build/outputs/apk/debug/phone-debug.apk`.
   - **List the rear cameras and test each with the torch on.** On Ultra and Honor camera bumps the
     main lens and the flash may not fit under one fingertip.
   - Then run a fingertip and a face recording, each with a simultaneous watch ECG or `hr_raw`.
5. **Phase 1 (no SDK), the nightly pipeline:** lowest 5-minute resting HR, motion-gated RMSSD/SDNN
   as an age/sex percentile (Natarajan 2020 tables), nightly respiratory rate, sleep/wake, personal
   baselines with NightSignal-style flags. Gate: agreement with the Samsung export.
6. **Phase 2:** run the AF screen on our own nights to measure the false-alert rate. Then SDK ECG
   confirmation with NeuroKit2 and ECGFounder (1-lead checkpoint; check the 150-class head loads
   with `strict=True`).
7. **CGM track with Livia** (`docs/cgm-track-plan.md`). Steps 1, 3 and 4 can be built on the
   historical file `~/data/Livia/OUTPUT/livia_glucose_insulin.parquet` before any watch nights.

## Risks & Blockers

- **The devices are the blocker.** The watch's Wireless debugging keeps turning off, and the phone
  has never been paired. Nothing past Phase 0 can be verified without them.
- **`hr_raw` access is undocumented.** It works on this firmware only (One UI Watch 8), and Samsung's
  2024 blog says third-party apps get null. A firmware update could close it. It is not yet
  ECG-validated, and it hides lost samples.
- **SDK developer mode** is "for testing only". Distribution needs Samsung partner approval tied to
  the package and signing key. Licence clause 3.3 bars medical or diagnostic use.
- **Disk:** about 65 GB free on a 1.9 TB drive (97% full). PulseLM (15 GB) plus datasets would make
  it tight.
- **Licences:** FacePhys may not run on the server. PaPaGei and AnyPPG weights are unlicensed.
  pyPPG is GPL plus "academic only". rPPG-Toolbox forbids inferring health conditions. D-BETA,
  ECG-Scan, Q-HEART and CP-PPG are CC BY-NC.
- **The server has no authentication** and is open on the LAN. Add a token before any non-home
  network.
- **Known minor issues the reviewers left open:**
  - **Phone app:**
    - the fingertip guide box is drawn over the cropped preview while the ROI is measured on the
      full frame (cosmetic);
    - `--ez retry true` is ignored while the camera is still opening;
    - a trace rejected with a 4xx stays pending and is re-sent on every retry;
    - a failed `error.txt` write can crash the I/O thread (`TraceStore.upload`).
  - **Watch app:**
    - a Health Platform connection that answers after the 10 s timeout is never closed;
    - a run whose SDK policy fails mid-run still says tier "all" in its `capabilities.json`, with
      the failed sources noted;
    - the health-type foreground service throws if no health permission is granted yet.
  - **Server:** Samsung IBI dropped-beat detection is tuned on synthetic data only, catching about
    60% of injected drops with 6 false alarms over 20 seeds.

## Open Questions

- Does the watch's `hr_raw` time beats as well as GalaxyPPG's SDK stream? That needs the ECG check
  above.
- Can SDK ECG_ON_DEMAND run together with SensorManager `hr_raw`? And PPG_ON_DEMAND together with
  continuous `hr_raw`?
- Which phone lens works best with the torch, what frame rate is real (30 vs 60 fps), and does
  exposure stay locked with the torch on? Tested only on the emulator.
- Did the user revoke the HF token pasted on 2026-10-21? Did they request access to PulseLM, D-BETA,
  ECG-Scan, CP-PPG, the PulseLM dataset and Meta's Llama-3.2-1B? These are low priority now, since
  those models proved of little value.
- Is the Samsung Health ECG PDF vector or raster? If vector, exact samples can be extracted.
- Questions for Livia from the CGM plan: which watch, whether she uses Samsung Health, which CGM and
  insulin data GlucoWatch reads, and how meals are logged.

## Quick Start for Next Session

```bash
# Restore context
cd ~/sources/pulse-exploration
cat plans/handoffs/HANDOFF_ppg-prototype-resume_2026-10-26.md
cat "reports/PPG health and aging measures.md" | sed -n '133,164p'   # build plan + post-report updates
cat ~/.claude/projects/-home-antonkulaga/memory/digital-health-ppg-prototype.md

# Key files to read first
#   pulse_exploration/watch_session.py (analysis, HRV priority, HRV25_BEST)
#   watch/probe/src/main/java/io/github/antonkulaga/pulseprobe/{RecorderService,Capabilities,Sources}.kt
#   pulse_exploration/hrv/cand_foot_template.py, pulse_exploration/hrv/af_screen.py
#   docs/hrv-25hz-benchmark.md, docs/phase0-offline-findings.md

# Verify current state
uv run pytest -q
(cd watch && timeout 590 ./gradlew --no-daemon -q :probe:assembleDebug :phone:assembleDebug)

# Start the server (background) and check it
uv run uvicorn pulse_exploration.server:app --host 0.0.0.0 --port 8765 &
curl -s http://127.0.0.1:8765/health && curl -s http://127.0.0.1:8765/runs/20261022-015638-record/analysis?refresh=1 | head -c 600

# Find and connect the watch (user must turn on Wireless debugging first; IP 192.168.1.222)
(timeout 8 stdbuf -oL avahi-browse -rp _adb-tls-connect._tcp 2>/dev/null; true) | grep '^=' | grep IPv4 | cut -d';' -f8,9
adb connect 192.168.1.222:<port>
A="adb -s 192.168.1.222:<port>"; PKG=io.github.antonkulaga.pulseprobe
$A install -r watch/probe/build/outputs/apk/debug/probe-debug.apk
for p in android.permission.health.READ_HEART_RATE android.permission.ACTIVITY_RECOGNITION \
  com.samsung.android.hardware.sensormanager.permission.READ_ADDITIONAL_HEALTH_DATA \
  android.permission.health.READ_SKIN_TEMPERATURE android.permission.health.READ_OXYGEN_SATURATION; do $A shell pm grant $PKG $p; done
$A shell input keyevent KEYCODE_WAKEUP
$A shell am start -S -f 0x10008000 -n $PKG/.MainActivity --es cmd wifi --ei seconds 3600 --es server http://192.168.1.177:8765
$A shell am start -n $PKG/.MainActivity --es cmd caps        # then: --es cmd spot --es feature ecg
$A logcat -d | grep PulseProbe | tail -20

# Phone emulator (only if no real phone): read-only, software GPU, host = 10.0.2.2
~/Android/Sdk/emulator/emulator -avd glucowatch_phone -read-only -no-snapshot -no-audio -no-boot-anim \
  -gpu swiftshader_indirect -cores 2 -no-window -camera-back emulated -camera-front emulated -port 5560 &
adb -s emulator-5560 shell am start -S -n io.github.antonkulaga.pulsephone/.MainActivity \
  --es mode finger --ei seconds 25 --es server http://10.0.2.2:8765 --ez autostart true

# Next action
# Ask the user to re-enable Wireless debugging on the watch (and answer the `adb tcpip 5555` question),
# install the capability-aware probe, then record the 30 s SDK ECG with hr_raw running (Phase 0 check).
```

### Environment notes for the next session
- **Laptop:** Linux, 32 cores, 94 GB RAM, RTX 4090 Laptop 16 GB, LAN IP 192.168.1.177. Watch on
  192.168.1.222. uv 0.10.0, Python 3.13 in the project.
- **Java:** no system `javac`. Gradle uses `~/.gradle/jdks/eclipse_adoptium-21-amd64-linux.2`.
  `javap` from there reads the SDK's API (`samsung-health-sensor-api-1.4.1.aar`, classes under
  `com.samsung.android.service.health.tracking`).
- **Android SDK** at `~/Android/Sdk` (platform 36, build-tools 35/36). AVDs: glucowatch_phone and
  several glucowatch watch images. Always run the emulator **read-only** with software graphics.
- **Commands that hang:** adb commands against a dropped watch block; wrap them in `timeout`.
- **open-rppg** needs `setuptools<82` (pinned). Importing it changes the server's global
  warning/logging settings.
- **`/capture` (HTTPS web page):** `uv run python scripts/make_cert.py` writes certificates to
  `data/certs/`. Serve on port 8766 with `--ssl-keyfile`/`--ssl-certfile`.

## Session Closed
**Closed at:** 2026-10-26T18:12:32+02:00
**Commit:** the `session: ppg-prototype-resume` commit that includes this note
**Session status:** Handed off to next session
