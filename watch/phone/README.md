# Pulse Phone (`:phone`)

Camera PPG on an ordinary phone, for prototyping only. The app averages the colour of a region of each
camera frame and sends those per-frame means (never the video) to the laptop server's
`POST /camera/trace`. It then shows the server's answer: HR, quality, tier and warnings.

- **Fingertip**: rear main camera with the torch on. Cover the lens and the flash with a fingertip. The
  region is the central half of the frame in each direction.
- **Face**: front camera. Put your forehead and upper cheeks inside the on-screen box. The region is that
  box, mapped into the analysis frame through a shared ViewPort.

A run goes: bind the camera → first frame → 2 s settling (AE/AWB free, torch on) → lock AE and AWB →
record N seconds on the sensor clock (default 45, range 20-120) → save locally → upload.

## Build and install

```sh
cd watch
./gradlew --no-daemon -q :phone:assembleDebug
adb install -g phone/build/outputs/apk/debug/phone-debug.apk   # -g grants the camera permission
```

If a watch is also connected, add `-s <serial>` to every `adb` command.

Start the server on the laptop so the phone can reach it over the LAN:

```sh
uv run uvicorn pulse_exploration.server:app --host 0.0.0.0 --port 8765
```

## Run from adb

```sh
adb shell am start -n io.github.antonkulaga.pulsephone/.MainActivity \
  --es mode finger --ei seconds 45 --es server http://192.168.1.177:8765 --ez autostart true
adb logcat -s PulsePhone          # applied settings, trace summary, result or upload error
```

| extra | meaning |
|---|---|
| `--es mode finger\|face` | mode |
| `--ei seconds N` | duration, clamped to 20-120 |
| `--es server URL` | server base URL |
| `--ez autostart true` | start recording at once |
| `--ez fps60 true` | ask for a fixed (60,60) AE range first |
| `--ei exposure_us N --ei iso N` | manual exposure (turns manual on) |
| `--ez manual false` | back to auto exposure |
| `--ez retry true` | re-send every trace the server has not answered |

Every extra is saved as the new default, so later runs from the UI use the same values.

Traces are stored in `/sdcard/Android/data/io.github.antonkulaga.pulsephone/files/traces/<stamp>-<mode>/`:

- `trace.json` is exactly the POST body.
- `frames.csv` holds per frame: clipping, exposure, ISO, frame duration and AE/AWB state, joined by sensor
  timestamp.
- `result.json` is the server's answer.
- `error.txt` holds the last upload error.

A trace without `result.json` is pending: use **Retry upload**, or `--ez retry true`. To copy the traces off
the phone: `adb pull /sdcard/Android/data/io.github.antonkulaga.pulsephone/files/traces`.

## What is sent

```json
{"mode": "finger", "t_ms": [0, 33.44, ...], "r": [170.28, ...], "g": [...], "b": [...],
 "meta": {"source": "pulsephone", "mode": "finger", "duration_s_requested": 45, "duration_s_recorded": 45.02,
          "stopped_early": false, "recorded_utc": "...", "app": {...}, "device": {...},
          "camera": {"camera_id", "hardware_level", "capabilities", "fps_ranges", "timestamp_source", ...},
          "applied": {"fps_range", "noise_reduction", "edge", "video_stabilization", "optical_stabilization",
                      "af_mode", "exposure", "ae_lock", "awb_lock", "torch", "settle_ms", "lock_confirmed", ...},
          "capture_result": {"exposure_ns": {min,median,max,distinct}, "iso": {...}, "frame_duration_ns": {...},
                             "ae_state": {"LOCKED": n}, "awb_state": {...}, "echo": {...}},
          "timing": {"t0_sensor_ns", "measured_fps", "frame_gaps", "frames_captured",
                     "frames_dropped_by_analysis", "frames_matched_to_capture_result", ...},
          "roi": {...}, "clipped_fraction_any": 0.0, "clipped_fraction_r": 0.0, "clipped_fraction_g": 0.0,
          "clipped_fraction_b": 0.0}}
```

- `t_ms` is the sensor timestamp (`ImageInfo.timestamp`) relative to the first frame. The origin is in
  `meta.timing.t0_sensor_ns`.
- `r`/`g`/`b` are region means in 0..255.
- A pixel counts as clipped at 250 or more. The server reads the `clipped_fraction_*` keys.
- Meta stays a few KB, well under the server's 1 MB cap (`MAX_META_BYTES` in camera_api.py). Per-frame details are only in `frames.csv`.

## Camera controls, and which are best-effort

The analysis stream is CameraX `ImageAnalysis` with `STRATEGY_KEEP_ONLY_LATEST` and
`OUTPUT_IMAGE_FORMAT_RGBA_8888`, at 640x480 (or the closest size the camera offers). The app has a YUV
fallback in case a device delivers YUV anyway. It never uses high-speed (120/240 fps) sessions: they force
auto exposure and deliver no analysis frames.

| control | how | per device |
|---|---|---|
| fixed AE fps range | Camera2Interop `CONTROL_AE_TARGET_FPS_RANGE`. With the 60 fps toggle on: (60,60) if listed. Otherwise (30,30), else the range topping out at 30 with the highest floor. | Samsung gives third-party apps about 30 fps, and 60 only sometimes. If no frame arrives at 60 within 5 s, the app falls back to 30 once. `timing.measured_fps` is what really happened. |
| torch | `CameraControl.enableTorch(true)`, fingertip mode only | needs a flash unit (`applied.torch`) |
| AE + AWB lock after 2 s | `Camera2CameraControl` `CONTROL_AE_LOCK` / `CONTROL_AWB_LOCK`, confirmed from capture results | only where `CONTROL_*_LOCK_AVAILABLE`. `applied.lock_confirmed` says whether a result showed the locks within 1.5 s. |
| manual exposure / ISO | Camera2Interop `CONTROL_AE_MODE_OFF` + `SENSOR_EXPOSURE_TIME` + `SENSOR_SENSITIVITY` + `SENSOR_FRAME_DURATION` (1/fps), clamped to the sensor ranges | offered only if the camera lists `MANUAL_SENSOR`, otherwise auto + lock |
| video stabilisation off | `CONTROL_VIDEO_STABILIZATION_MODE_OFF` | if listed |
| optical stabilisation off | `LENS_OPTICAL_STABILIZATION_MODE_OFF` | if listed |
| noise reduction off | `NOISE_REDUCTION_MODE_OFF`, else `MINIMAL` (temporal NR would smear the pulse) | if listed |
| edge enhancement off | `EDGE_MODE_OFF` | if listed |
| AF off (fingertip) | `CONTROL_AF_MODE_OFF`, so the lens stops hunting under the finger | if listed |

What was requested is in `meta.applied`. What the HAL echoed back is in `meta.capture_result.echo`. It is
read from the session capture callback, which also supplies the per-frame exposure, ISO, frame duration and
AE state.

On some devices these settings may not hold: listed modes can be ignored, AE lock can drift, and 60 fps can
be listed but not delivered. Compare `applied` with `capture_result` per device.

Implementation notes (CameraX 1.6.2 runs on the CameraPipe backend):

- Interop options are `ALWAYS_OVERRIDE`. AE/AF modes from them become CameraPipe's 3A state. The lock keys
  are kept out of interop so that two `ALWAYS_OVERRIDE` sources never fight over one key.
- Switching the torch off makes CameraX reset AE to ON for the rest of the session. So after a
  manual-exposure fingertip run, the camera is re-bound before the next run.

## Limits

- Portrait only. On screens with a smallest width of 600 dp or more (tablets, unfolded foldables),
  Android 16 ignores the orientation lock. The ROI mapping assumes portrait.
- Cleartext HTTP is allowed to every host (`res/xml/network_security_config.xml`), because the server
  address is a user-editable LAN IP. This is for the prototype only.
- Traces are personal health data. They are excluded from backup and device transfer, and you have to
  delete them on the phone yourself.

Dependencies:

| library | version |
|---|---|
| `androidx.camera:camera-core`, `-camera2`, `-lifecycle`, `-view` | 1.6.2 |
| `androidx.activity:activity` | 1.13.0 |
| `androidx.core:core` | 1.18.0 (1.19 needs compileSdk 37 and AGP 9.1) |

HTTP uses `HttpURLConnection`, JSON uses `org.json`, and the views are plain Android Views.
