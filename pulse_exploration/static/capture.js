/* Tier "capture" page: the page drives the camera with getUserMedia, turns the torch on (fingertip),
 * locks exposure / white balance / focus when the browser offers it, and records one mean R, G, B per
 * video frame over a region of interest. After the countdown it POSTs
 *   {mode, t_ms, r, g, b, meta}
 * to /camera/trace. The video itself never leaves the phone.
 *
 * URL options for testing: ?synthetic=1 replaces the camera with a generated pulsing colour
 * (marked synthetic in meta), &bpm=72 sets its rate, &seconds=45 sets the duration (10..120).
 */
(function () {
  "use strict";
  const UI = window.PulseUI;
  const SIG = window.PulseSignal;
  const $ = (id) => document.getElementById(id);
  const body = document.body;
  const params = new URLSearchParams(location.search);
  const ENDPOINT = body.dataset.endpoint || "/camera/trace";
  const SYNTHETIC = params.get("synthetic") === "1";
  const DEFAULT_S = Number(body.dataset.durationS) || 45;
  const URL_S = Number(params.get("seconds")) || null;
  let DURATION_S = DEFAULT_S;
  const MIN_FRAMES = 100;
  const EARLY_FINISH_S = 20;

  // Regions of interest, as fractions of the video frame. The face box is centred horizontally, so it is
  // the same region whether or not the preview is mirrored.
  const ROI = {
    finger: { kind: "center", x: 0.25, y: 0.25, w: 0.5, h: 0.5 },
    face: { kind: "forehead", x: 0.37, y: 0.16, w: 0.26, h: 0.11 },
  };
  const LOCKS = [
    { label: "exposure", modeKey: "exposureMode", valueKeys: ["exposureTime", "iso"] },
    { label: "white balance", modeKey: "whiteBalanceMode", valueKeys: ["colorTemperature"] },
    { label: "focus", modeKey: "focusMode", valueKeys: ["focusDistance"] },
  ];

  const video = $("video");
  const startCamBtn = $("start-cam");
  const startRecBtn = $("start-rec");
  const sampler = document.createElement("canvas");
  const sctx = sampler.getContext("2d", { willReadFrequently: true });

  let phase = "off"; // off | starting | ready | locking | recording | sending
  let stream = null;
  let track = null;
  let synth = null;
  let caps = {};
  let gumInfo = null;
  let camMode = "finger"; // mode the running camera was opened for
  let deviceChoice = null;
  let requestedFps = 30;
  let controls = []; // results of every control we tried, in order
  let imageSets = []; // advanced constraint sets that applied, kept so later calls do not undo them
  let settingsStart = null;
  let settingsBeforeLock = null;
  let settingsAfterLock = null;
  let loopKind = null;
  let loopGen = 0;
  let frames = [];
  let recStart = 0;
  let recTimer = 0;
  let startedAt = null;
  let wakeLock = null;
  let lastPayload = null;
  const live = { t: [], v: [], fpsT: [], lastPF: null, missed: 0, lastDraw: 0, lastText: 0, fps: null };

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function mode() {
    const c = document.querySelector('input[name="mode"]:checked');
    return c ? c.value : "finger";
  }

  function setCamStatus(text, isError) {
    const el = $("cam-status");
    el.textContent = text || "";
    el.classList.toggle("error", !!isError);
  }

  function setRecStatus(text, isError) {
    const el = $("rec-status");
    el.textContent = text || "";
    el.classList.toggle("error", !!isError);
  }

  function setPhase(p) {
    phase = p;
    const camOn = !!track;
    const busy = p === "starting" || p === "locking" || p === "recording" || p === "sending";
    startCamBtn.disabled = busy || !canUseCamera();
    startCamBtn.textContent = camOn ? "Restart camera" : "Start camera";
    startRecBtn.disabled = p !== "ready";
    startRecBtn.hidden = p === "locking" || p === "recording";
    $("recording").hidden = !(p === "locking" || p === "recording");
    document.querySelectorAll('input[name="mode"]').forEach((r) => { r.disabled = busy; });
    $("fps").disabled = busy;
    $("duration").disabled = busy;
    $("camera").disabled = busy;
    $("stage").hidden = !camOn;
    $("live-stats").hidden = !camOn;
    $("live-plot").hidden = !camOn;
    $("coverage").hidden = !camOn;
    $("controls-box").hidden = !camOn && controls.length === 0;
  }

  function canUseCamera() {
    if (SYNTHETIC) return true;
    return !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  function cleanSettings(s) {
    if (!s) return null;
    const out = Object.assign({}, s);
    delete out.deviceId; // per-origin random ids: not useful for analysis
    delete out.groupId;
    return out;
  }

  function getCaps(t) {
    try {
      const c = t.getCapabilities ? t.getCapabilities() : {};
      return cleanSettings(c) || {};
    } catch (e) {
      return {};
    }
  }

  function errText(e) {
    if (!e) return "unknown error";
    return (e.name ? e.name + ": " : "") + (e.message || String(e));
  }

  // ---------- camera start / stop ----------

  function synthStream(m, fps) {
    const c = document.createElement("canvas");
    c.width = 480;
    c.height = 640;
    const ctx = c.getContext("2d");
    const bpm = Number(params.get("bpm")) || 72;
    const t0 = performance.now();
    const draw = () => {
      const t = (performance.now() - t0) / 1000;
      const ph = 2 * Math.PI * (bpm / 60) * t;
      const p = Math.sin(ph) + 0.35 * Math.sin(2 * ph + 0.8);
      const br = Math.sin(2 * Math.PI * 0.25 * t);
      const n = () => Math.random() - 0.5;
      let r, g, b;
      if (m === "finger") {
        r = 205 - 6 * p + 2 * br + n();
        g = 40 - 2.5 * p + n();
        b = 22 + 0.5 * n();
      } else {
        r = 175 - 1.0 * p + n();
        g = 120 - 2.0 * p + n();
        b = 95 - 0.5 * p + n();
      }
      ctx.fillStyle = "rgb(" + r.toFixed(2) + "," + g.toFixed(2) + "," + b.toFixed(2) + ")";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 28px sans-serif";
      ctx.fillText("SYNTHETIC " + bpm + " bpm", 20, c.height - 30); // below both ROIs
    };
    draw();
    const id = setInterval(draw, 1000 / fps);
    const s = c.captureStream(fps);
    synth = { stop: () => clearInterval(id) };
    return s;
  }

  async function startCamera() {
    if (!canUseCamera()) return;
    stopCamera();
    setPhase("starting");
    setCamStatus("Starting camera…");
    camMode = mode();
    requestedFps = Number($("fps").value) || 30;
    const v = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: requestedFps } };
    if (deviceChoice) v.deviceId = { exact: deviceChoice };
    else v.facingMode = { ideal: camMode === "face" ? "user" : "environment" };
    gumInfo = { video: v, retried_without_size_and_fps: false };
    try {
      if (SYNTHETIC) {
        stream = synthStream(camMode, requestedFps);
      } else {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: v });
        } catch (e) {
          if (e.name !== "OverconstrainedError") throw e;
          const v2 = deviceChoice ? { deviceId: { exact: deviceChoice } } : { facingMode: v.facingMode };
          gumInfo = { video: v2, retried_without_size_and_fps: true, first_error: errText(e) };
          stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: v2 });
        }
      }
    } catch (e) {
      stream = null;
      setPhase("off");
      setCamStatus(cameraErrorMessage(e), true);
      return;
    }
    track = stream.getVideoTracks()[0];
    track.addEventListener("ended", () => {
      if (phase === "recording" || phase === "locking") abortRecording("The camera stopped during the measurement.");
      stopCamera();
      setCamStatus("The camera stopped. Tap Start camera to try again.", true);
    });
    video.srcObject = stream;
    $("stage").dataset.mode = camMode;
    try { await video.play(); } catch (e) { /* muted inline video normally autoplays */ }
    controls = [];
    imageSets = [];
    settingsBeforeLock = null;
    settingsAfterLock = null;
    live.t = []; live.v = []; live.fpsT = []; live.lastPF = null; live.missed = 0; live.fps = null;
    startLoop();
    setPhase("starting");
    const mine = track;
    await sleep(400); // capabilities can lag behind the first frames
    if (track !== mine) return; // stopped or restarted meanwhile
    caps = await readCapabilities(track, camMode);
    if (track !== mine) return;
    settingsStart = cleanSettings(track.getSettings());
    if (camMode === "finger") controls.push(await setTorch(true));
    if (track !== mine) return;
    renderControls();
    listCameras();
    setPhase("ready");
    setCamStatus(cameraSummary());
  }

  function cameraErrorMessage(e) {
    switch (e && e.name) {
      case "NotAllowedError":
        return "Camera permission was refused. Tap the icon left of the address, allow Camera, then try again.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No matching camera was found. Try the other mode or another camera.";
      case "NotReadableError":
        return "The camera is busy. Close other apps that use it (camera, video calls) and try again.";
      case "SecurityError":
        return "Camera blocked: open this page over HTTPS.";
      default:
        return "Could not start the camera: " + errText(e);
    }
  }

  async function readCapabilities(t, m) {
    let c = getCaps(t);
    if (m === "finger" && !("torch" in c) && typeof ImageCapture !== "undefined") {
      // Some Chrome builds only report torch after the image-capture path has been queried once.
      try { await new ImageCapture(t).getPhotoCapabilities(); } catch (e) { /* not supported: fine */ }
      await sleep(300);
      c = getCaps(t);
    }
    return c;
  }

  function stopCamera() {
    loopGen++;
    if (stream) stream.getTracks().forEach((t) => t.stop()); // stopping the track also turns the torch off
    if (synth) synth.stop();
    synth = null;
    stream = null;
    track = null;
    video.srcObject = null;
    if (phase === "ready" || phase === "starting") setPhase("off");
  }

  async function listCameras() {
    if (SYNTHETIC || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      const sel = $("camera");
      const current = track ? track.getSettings().deviceId : null;
      sel.innerHTML = "";
      devs.forEach((d, i) => {
        const o = document.createElement("option");
        o.value = d.deviceId;
        o.textContent = d.label || "Camera " + (i + 1);
        if (d.deviceId === current) o.selected = true;
        sel.appendChild(o);
      });
      $("camera-row").hidden = devs.length < 2;
    } catch (e) { /* the picker is optional */ }
  }

  // ---------- camera controls ----------

  function sameValue(a, b) {
    if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= Math.max(1e-6, Math.abs(b) * 0.02);
    return a === b;
  }

  async function tryControl(label, set) {
    const res = { control: label, requested: set };
    if (!track) return Object.assign(res, { status: "error", error: "no camera" });
    try {
      await track.applyConstraints({ advanced: imageSets.concat([set]) });
    } catch (e) {
      return Object.assign(res, { status: "error", error: errText(e) });
    }
    await sleep(200);
    const st = track.getSettings();
    res.reported = {};
    Object.keys(set).forEach((k) => { res.reported[k] = st[k] === undefined ? null : st[k]; });
    res.status = Object.keys(set).every((k) => sameValue(st[k], set[k])) ? "applied" : "ignored";
    if (res.status === "applied") imageSets.push(set);
    return res;
  }

  async function setTorch(on) {
    // Chrome reports torch as true; the spec once allowed a list of supported values.
    const offered = caps.torch === true || (Array.isArray(caps.torch) && caps.torch.includes(true));
    if (!offered) return { control: "torch", requested: { torch: on }, status: "not offered" };
    return tryControl("torch", { torch: on });
  }

  function clampToRange(v, range) {
    if (!range || typeof range !== "object") return v;
    let x = v;
    if (typeof range.min === "number") x = Math.max(range.min, x);
    if (typeof range.max === "number") x = Math.min(range.max, x);
    return x;
  }

  /** Freeze one auto control at its current value: 'manual' with the current value, else 'single-shot'. */
  async function lockOne(spec, st) {
    const modes = Array.isArray(caps[spec.modeKey]) ? caps[spec.modeKey] : [];
    const single = {};
    single[spec.modeKey] = "single-shot";
    if (modes.includes("manual")) {
      const set = {};
      set[spec.modeKey] = "manual";
      spec.valueKeys.forEach((k) => {
        if (caps[k] && typeof st[k] === "number") set[k] = clampToRange(st[k], caps[k]);
      });
      const r = await tryControl(spec.label, set);
      if (r.status === "applied" || !modes.includes("single-shot")) return r;
      const r2 = await tryControl(spec.label, single);
      r2.fallback_from = r;
      return r2;
    }
    if (modes.includes("single-shot")) return tryControl(spec.label, single);
    return { control: spec.label, status: "not offered", offered_modes: modes };
  }

  async function lockControls() {
    const out = [];
    for (const spec of LOCKS) {
      if (phase !== "locking") break;
      out.push(await lockOne(spec, track ? track.getSettings() : {}));
    }
    return out;
  }

  function compact(obj) {
    if (obj === undefined || obj === null) return "";
    try { return JSON.stringify(obj).replace(/[{}"]/g, "").replace(/,/g, ", ").replace(/:/g, ": "); } catch (e) { return ""; }
  }

  function renderControls() {
    const rows = [];
    const badge = { applied: "ok", ignored: "warn", error: "bad", "not offered": "plain" };
    controls.forEach((c) => {
      const detail = [];
      if (c.requested) detail.push("asked " + compact(c.requested));
      if (c.reported) detail.push("camera reports " + compact(c.reported));
      if (c.error) detail.push(c.error);
      if (c.fallback_from) detail.push("manual was " + c.fallback_from.status + ", used single-shot");
      rows.push("<tr><th>" + UI.escapeHTML(c.control) + '</th><td><span class="badge ' + (badge[c.status] || "plain") + '">' +
        UI.escapeHTML(c.status) + '</span> <span class="muted">' + UI.escapeHTML(detail.join(" · ")) + "</span></td></tr>");
    });
    const tried = new Set(controls.map((c) => c.control));
    LOCKS.forEach((spec) => {
      if (tried.has(spec.label)) return;
      const modes = Array.isArray(caps[spec.modeKey]) ? caps[spec.modeKey] : [];
      const lockable = modes.includes("manual") || modes.includes("single-shot");
      rows.push("<tr><th>" + UI.escapeHTML(spec.label) + "</th><td>" + (lockable
        ? '<span class="badge plain">locks at start</span> <span class="muted">offered: ' + UI.escapeHTML(modes.join(", ")) + "</span>"
        : '<span class="badge plain">not offered</span>') + "</td></tr>");
    });
    const st = track ? track.getSettings() : settingsAfterLock || settingsStart || {};
    if (st.width) rows.push("<tr><th>resolution</th><td>" + st.width + "×" + st.height + "</td></tr>");
    const fr = caps.frameRate;
    if (fr && typeof fr.max === "number") {
      rows.push("<tr><th>frame rate</th><td>asked " + requestedFps + ", camera reports " +
        (typeof st.frameRate === "number" ? UI.fmtNum(st.frameRate) : "?") + ", offers up to " + UI.fmtNum(fr.max) + "</td></tr>");
    }
    if (track && track.label) rows.push("<tr><th>camera</th><td>" + UI.escapeHTML(track.label) + "</td></tr>");
    $("controls").querySelector("tbody").innerHTML = rows.join("");
    $("controls-box").hidden = rows.length === 0;
  }

  function cameraSummary() {
    const parts = [];
    const torch = controls.find((c) => c.control === "torch");
    if (camMode === "finger") {
      parts.push(torch && torch.status === "applied" ? "Torch on." : "Torch not available on this camera: use another camera or a bright lamp.");
    }
    const lockable = LOCKS.filter((s) => {
      const m = Array.isArray(caps[s.modeKey]) ? caps[s.modeKey] : [];
      return m.includes("manual") || m.includes("single-shot");
    }).map((s) => s.label);
    parts.push(lockable.length ? "Can lock: " + lockable.join(", ") + "." : "This browser offers no exposure lock here; results may drift.");
    if (requestedFps === 60 && caps.frameRate && caps.frameRate.max < 60) {
      parts.push("This camera offers at most " + UI.fmtNum(caps.frameRate.max) + " fps to the browser.");
    }
    return parts.join(" ");
  }

  // ---------- per-frame sampling ----------

  function startLoop() {
    const gen = ++loopGen;
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      loopKind = "requestVideoFrameCallback";
      const cb = (now, md) => {
        if (gen !== loopGen) return;
        video.requestVideoFrameCallback(cb);
        onFrame(now, md || {});
      };
      video.requestVideoFrameCallback(cb);
    } else {
      // Fallback: poll every display frame and keep only new video frames.
      loopKind = "requestAnimationFrame";
      let lastCount = -1;
      let lastTime = -1;
      const cb = (now) => {
        if (gen !== loopGen) return;
        requestAnimationFrame(cb);
        const q = video.getVideoPlaybackQuality ? video.getVideoPlaybackQuality() : null;
        const count = q ? q.totalVideoFrames : -1;
        if (count >= 0 ? count === lastCount : video.currentTime === lastTime) return;
        lastCount = count;
        lastTime = video.currentTime;
        onFrame(now, { presentedFrames: count >= 0 ? count : undefined });
      };
      requestAnimationFrame(cb);
    }
  }

  function onFrame(now, md) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;
    const roi = ROI[camMode];
    const sx = Math.round(roi.x * vw);
    const sy = Math.round(roi.y * vh);
    const sw = Math.max(1, Math.round(roi.w * vw));
    const sh = Math.max(1, Math.round(roi.h * vh));
    const cw = Math.min(sw, 128);
    const ch = Math.min(sh, 128);
    if (sampler.width !== cw || sampler.height !== ch) {
      sampler.width = cw;
      sampler.height = ch;
    }
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = "high";
    let s;
    try {
      sctx.drawImage(video, sx, sy, sw, sh, 0, 0, cw, ch);
      s = SIG.roiStats(sctx.getImageData(0, 0, cw, ch).data);
    } catch (e) {
      return;
    }
    if (!s) return;
    const num = (x) => (typeof x === "number" && Number.isFinite(x) ? x : null);
    const frame = {
      mediaTime: num(md.mediaTime) !== null ? md.mediaTime * 1000 : null,
      captureTime: num(md.captureTime),
      expectedDisplayTime: num(md.expectedDisplayTime),
      now: now,
      presentedFrames: num(md.presentedFrames),
      r: s.r,
      g: s.g,
      b: s.b,
      sat: s.sat,
    };
    updateLive(frame);
    if (phase === "recording") frames.push(frame);
  }

  function updateLive(f) {
    live.fpsT.push(f.now);
    while (live.fpsT.length > 2 && f.now - live.fpsT[0] > 2000) live.fpsT.shift();
    if (live.fpsT.length > 2) {
      const span = (live.fpsT[live.fpsT.length - 1] - live.fpsT[0]) / 1000;
      live.fps = span > 0 ? (live.fpsT.length - 1) / span : null;
    }
    if (f.presentedFrames !== null && live.lastPF !== null) live.missed += Math.max(0, f.presentedFrames - live.lastPF - 1);
    live.lastPF = f.presentedFrames;
    const value = camMode === "finger" ? (f.r < 250 ? f.r : f.g) : f.g;
    live.t.push(f.now);
    live.v.push(value);
    while (live.t.length && f.now - live.t[0] > 8000) { live.t.shift(); live.v.shift(); }

    if (f.now - live.lastText > 250) {
      live.lastText = f.now;
      $("live-fps").textContent = live.fps ? live.fps.toFixed(1) : "–";
      $("live-r").textContent = f.r.toFixed(0);
      $("live-g").textContent = f.g.toFixed(0);
      $("live-b").textContent = f.b.toFixed(0);
      $("live-missed").textContent = String(live.missed);
      const [cls, msg] = coverage(f);
      const el = $("coverage");
      el.className = "notice " + cls;
      el.textContent = msg;
    }
    if (f.now - live.lastDraw > 66 && live.v.length > 10) {
      live.lastDraw = f.now;
      const win = Math.max(5, Math.round(live.fps || 30));
      // Detrend and invert: more blood absorbs more light, so pulses point up.
      const y = SIG.detrend(live.v, win).map((x) => -x);
      UI.drawWaveform($("live-plot"), { y: y }, { lineWidth: 2 });
    }
  }

  function coverage(f) {
    if (camMode === "finger") {
      const torch = controls.find((c) => c.control === "torch");
      const torchOn = torch && torch.status === "applied";
      if (f.r < 40) {
        return ["bad", torchOn ? "Too dark: cover the lens and the torch with your fingertip."
          : "Too dark and the torch is off: hold a bright lamp behind your finger, or pick another camera."];
      }
      if (f.r > 1.8 * f.g && f.r > 1.8 * f.b && f.r > 60) {
        if (f.sat > 0.9) return ["warn", "Finger detected, image very bright. Usually fine; press a little lighter if the trace looks flat."];
        return ["ok", "Finger detected. Keep still."];
      }
      return ["warn", "Cover the main lens fully with your fingertip, so the preview is evenly red."];
    }
    const y = 0.299 * f.r + 0.587 * f.g + 0.114 * f.b;
    if (y < 50) return ["bad", "Too dark: face a window or a lamp."];
    if (f.sat > 0.15) return ["warn", "Part of the box is overexposed: soften or move the light."];
    return ["ok", "Brightness OK. Keep your forehead inside the small box and stay still."];
  }

  // ---------- recording ----------

  async function acquireWakeLock() {
    try {
      if (navigator.wakeLock) wakeLock = await navigator.wakeLock.request("screen");
    } catch (e) {
      wakeLock = null;
    }
  }

  function releaseWakeLock() {
    try { if (wakeLock) wakeLock.release(); } catch (e) { /* already released */ }
    wakeLock = null;
  }

  async function startRecording() {
    if (!track || phase !== "ready") return;
    $("result").hidden = true;
    $("retry").hidden = true;
    lastPayload = null;
    setPhase("locking");
    $("countdown").textContent = String(DURATION_S);
    $("rec-bar").value = 0;
    $("finish-early").disabled = true;
    setRecStatus("Locking exposure and white balance…");
    settingsBeforeLock = cleanSettings(track.getSettings());
    const lockLabels = new Set(LOCKS.map((l) => l.label));
    controls = controls.filter((c) => !lockLabels.has(c.control)); // a cancelled run may have tried already
    const locks = await lockControls();
    if (phase !== "locking") return; // cancelled meanwhile
    controls = controls.concat(locks);
    const torch = controls.find((c) => c.control === "torch");
    if (camMode === "finger" && torch && torch.status === "applied" && track && track.getSettings().torch !== true) {
      const r = await setTorch(true);
      r.control = "torch (again after lock)";
      controls.push(r);
    }
    if (!track || phase !== "locking") return;
    settingsAfterLock = cleanSettings(track.getSettings());
    renderControls();
    await acquireWakeLock();
    setRecStatus("Settling…");
    await sleep(1000);
    if (!track || phase !== "locking") return;
    frames = [];
    startedAt = new Date().toISOString();
    recStart = performance.now();
    setPhase("recording");
    recTimer = setInterval(tick, 200);
    tick();
  }

  function tick() {
    const elapsed = (performance.now() - recStart) / 1000;
    const left = Math.max(0, DURATION_S - elapsed);
    $("countdown").textContent = String(Math.ceil(left));
    $("rec-bar").value = Math.min(100, (100 * elapsed) / DURATION_S);
    $("finish-early").disabled = elapsed < EARLY_FINISH_S;
    setRecStatus("Recording… " + frames.length + " frames. Keep still.");
    if (left <= 0) finishRecording();
  }

  function abortRecording(message, isError) {
    clearInterval(recTimer);
    recTimer = 0;
    frames = [];
    releaseWakeLock();
    setPhase(track ? "ready" : "off");
    setRecStatus(message || "", isError !== false);
  }

  async function uaData() {
    const d = navigator.userAgentData;
    if (!d) return null;
    const out = { brands: d.brands, mobile: d.mobile, platform: d.platform };
    try {
      const hi = await Promise.race([d.getHighEntropyValues(["model", "platformVersion"]), sleep(1000).then(() => null)]);
      if (hi) {
        out.model = hi.model;
        out.platformVersion = hi.platformVersion;
      }
    } catch (e) { /* optional */ }
    return out;
  }

  async function buildMeta(settingsEnd) {
    const roi = ROI[camMode];
    return {
      page: "capture",
      tier: "capture",
      synthetic: SYNTHETIC,
      user_agent: navigator.userAgent,
      ua_data: await uaData(),
      started_at: startedAt,
      duration_target_s: DURATION_S,
      requested_fps: requestedFps,
      constraints_requested: gumInfo,
      track_label: track ? track.label : null,
      track_settings: { start: settingsStart, before_lock: settingsBeforeLock, after_lock: settingsAfterLock, end: settingsEnd },
      capabilities: caps,
      applied_constraints: track && track.getConstraints ? track.getConstraints() : null,
      controls: controls,
      roi: { kind: roi.kind, x: roi.x, y: roi.y, w: roi.w, h: roi.h, sample_px: { w: sampler.width, h: sampler.height } },
      video_size: { w: video.videoWidth, h: video.videoHeight },
      frame_callback: loopKind,
      live_frames_missed: live.missed,
      screen: {
        w: screen.width,
        h: screen.height,
        dpr: window.devicePixelRatio || 1,
        orientation: screen.orientation ? screen.orientation.type : null,
      },
    };
  }

  async function finishRecording() {
    if (phase !== "recording") return;
    clearInterval(recTimer);
    recTimer = 0;
    const recorded = frames;
    frames = [];
    setPhase("sending");
    const settingsEnd = track ? cleanSettings(track.getSettings()) : null;
    const meta = await buildMeta(settingsEnd);
    stopCamera();
    releaseWakeLock();
    try { if (navigator.vibrate) navigator.vibrate(150); } catch (e) { /* optional */ }
    if (recorded.length < MIN_FRAMES) {
      setPhase("off");
      setRecStatus("Only " + recorded.length + " frames were recorded. Check that the preview moves and try again.", true);
      return;
    }
    lastPayload = SIG.buildTrace(recorded, camMode, meta);
    await sendTrace(lastPayload);
  }

  async function sendTrace(payload) {
    setPhase("sending");
    setRecStatus("Analysing on the laptop…");
    $("retry").hidden = true;
    try {
      const res = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const text = await res.text();
      if (!res.ok) {
        setRecStatus(UI.errorMessage(res.status, text, "POST " + ENDPOINT), true);
        $("retry").hidden = false;
        return;
      }
      let result;
      try {
        result = JSON.parse(text);
      } catch (e) {
        setRecStatus("The server answered, but not with JSON.", true);
        $("retry").hidden = false;
        return;
      }
      setRecStatus("Done. " + payload.t_ms.length + " frames at " +
        (payload.meta.measured_fps.median_fps || "?") + " fps were analysed.");
      UI.renderResult($("result"), result);
    } catch (e) {
      setRecStatus("Network error: " + errText(e) + ". The recording is kept on this page; tap Send again.", true);
      $("retry").hidden = false;
    } finally {
      setPhase(track ? "ready" : "off");
    }
  }

  // ---------- wiring ----------

  function applyMode() {
    const m = mode();
    document.querySelectorAll(".howto").forEach((el) => { el.hidden = el.dataset.for !== m; });
    UI.savePref("mode", m);
    if (!track) $("stage").dataset.mode = m;
  }

  function applyDuration() {
    DURATION_S = Math.min(120, Math.max(10, Number($("duration").value) || DEFAULT_S));
    $("duration-label").textContent = String(DURATION_S);
    $("countdown").textContent = String(DURATION_S);
  }

  function placeGuides() {
    [["face-roi", ROI.face], ["finger-roi", ROI.finger]].forEach(([id, r]) => {
      const el = $(id);
      el.style.left = r.x * 100 + "%";
      el.style.top = r.y * 100 + "%";
      el.style.width = r.w * 100 + "%";
      el.style.height = r.h * 100 + "%";
    });
  }

  document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => {
    applyMode();
    deviceChoice = null;
    if (track) startCamera();
  }));
  $("fps").addEventListener("change", () => { if (track) startCamera(); });
  $("duration").addEventListener("change", applyDuration);
  $("camera").addEventListener("change", (e) => { deviceChoice = e.target.value || null; startCamera(); });
  startCamBtn.addEventListener("click", startCamera);
  startRecBtn.addEventListener("click", startRecording);
  $("cancel-rec").addEventListener("click", () => abortRecording("Measurement cancelled.", false));
  $("finish-early").addEventListener("click", finishRecording);
  $("retry").addEventListener("click", () => { if (lastPayload) sendTrace(lastPayload); });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && (phase === "recording" || phase === "locking")) {
      abortRecording("Stopped: the page went to the background. Keep this page open during the measurement.");
    }
  });
  window.addEventListener("pagehide", stopCamera);

  const saved = new URLSearchParams(location.search).get("mode") || UI.loadPref("mode", "finger");
  const radio = document.querySelector('input[name="mode"][value="' + (saved === "face" ? "face" : "finger") + '"]');
  if (radio) radio.checked = true;
  if (URL_S) {
    const opt = document.createElement("option");
    opt.value = String(Math.min(120, Math.max(10, URL_S)));
    opt.textContent = opt.value + " s (from the address)";
    $("duration").appendChild(opt);
    $("duration").value = opt.value;
  }
  applyDuration();
  $("synthetic-banner").hidden = !SYNTHETIC;
  placeGuides();
  applyMode();
  setPhase("off");
  if (!canUseCamera()) {
    if (!window.isSecureContext) {
      $("insecure").hidden = false;
      $("cert-help").open = true;
      setCamStatus("The camera needs HTTPS. Use the link above.", true);
    } else {
      setCamStatus("This browser does not offer camera access to web pages.", true);
    }
  }
})();
