/* Tier "video" page: pick or record a video with the stock camera app, upload it to POST /camera/video
 * (multipart fields: file, mode) and show the result. No camera API; works over plain HTTP.
 * Uses XMLHttpRequest for the upload because fetch() reports no upload progress.
 */
(function () {
  "use strict";
  const UI = window.PulseUI;
  const ENDPOINT = document.body.dataset.endpoint || "/camera/video";
  const $ = (id) => document.getElementById(id);

  const record = $("record");
  const pick = $("pick");
  const send = $("send");
  const cancel = $("cancel");
  const statusEl = $("status");
  const resultEl = $("result");
  let file = null;
  let xhr = null;
  let timer = 0;

  function mode() {
    const checked = document.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : "finger";
  }

  function setStatus(text, isError) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("error", !!isError);
  }

  function applyMode() {
    const m = mode();
    // The capture attribute tells Chrome which camera to open; the camera app may still remember its last one.
    record.setAttribute("capture", m === "face" ? "user" : "environment");
    $("record-text").textContent = m === "face" ? "Record with the front camera" : "Record with the rear camera";
    document.querySelectorAll(".howto").forEach((el) => { el.hidden = el.dataset.for !== m; });
    UI.savePref("mode", m);
  }

  function describe(f) {
    const info = $("file-info");
    const warn = $("file-warn");
    info.hidden = false;
    info.textContent = (f.name || "video") + " · " + UI.fmtBytes(f.size) + (f.type ? " · " + f.type : "");
    const warnings = [];
    if (f.size > 400e6) warnings.push("Large file: the upload will take a while. 1080p is enough.");
    if (f.type && !f.type.startsWith("video/")) warnings.push("This does not look like a video file.");
    warn.hidden = warnings.length === 0;
    warn.textContent = warnings.join(" ");
    // Read duration and size locally (nothing is uploaded yet). Some codecs may not decode here; that is fine.
    const url = URL.createObjectURL(f);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.muted = true;
    const done = () => { URL.revokeObjectURL(url); v.removeAttribute("src"); v.load(); };
    v.onloadedmetadata = () => {
      const d = v.duration;
      if (Number.isFinite(d)) {
        info.textContent += " · " + d.toFixed(1) + " s" + (v.videoWidth ? " · " + v.videoWidth + "×" + v.videoHeight : "");
        if (d < 20) {
          warnings.push("Shorter than 20 s: the estimate will be unreliable. Aim for 30–60 s.");
        } else if (d > 120) {
          warnings.push("Longer than 2 minutes: 30–60 s is enough and uploads faster.");
        }
        warn.hidden = warnings.length === 0;
        warn.textContent = warnings.join(" ");
      }
      done();
    };
    v.onerror = done;
    v.src = url;
  }

  function choose(input) {
    const f = input.files && input.files[0];
    input.value = ""; // so picking the same file again still fires "change"
    if (!f) return;
    file = f;
    describe(f);
    send.disabled = false;
    setStatus("Ready to upload.");
  }

  function busy(on) {
    send.disabled = on || !file;
    record.disabled = on;
    pick.disabled = on;
    $("progress").hidden = !on;
    document.querySelectorAll('input[name="mode"]').forEach((r) => { r.disabled = on; });
  }

  function upload() {
    if (!file) return;
    const m = mode();
    const form = new FormData();
    form.append("mode", m);
    form.append("file", file, file.name || "video.mp4");
    resultEl.hidden = true;
    busy(true);
    const bar = $("bar");
    const text = $("progress-text");
    bar.value = 0;
    bar.max = 100;
    const started = performance.now();
    setStatus("Uploading…");
    xhr = new XMLHttpRequest();
    xhr.open("POST", ENDPOINT);
    xhr.timeout = 15 * 60 * 1000; // deep models can be slow on first use
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = (100 * e.loaded) / e.total;
      bar.value = pct;
      const secs = (performance.now() - started) / 1000;
      const rate = secs > 0 ? e.loaded / secs : 0;
      const eta = rate > 0 ? (e.total - e.loaded) / rate : 0;
      text.textContent = pct.toFixed(0) + "% · " + UI.fmtBytes(e.loaded) + " of " + UI.fmtBytes(e.total) +
        (rate > 0 ? " · " + UI.fmtBytes(rate) + "/s" + (eta > 1 ? " · " + Math.ceil(eta) + " s left" : "") : "");
    };
    xhr.upload.onload = () => {
      bar.removeAttribute("value"); // indeterminate while the laptop analyses
      const t0 = performance.now();
      const tick = () => {
        text.textContent = "Uploaded. Analysing on the laptop… " + Math.round((performance.now() - t0) / 1000) + " s";
      };
      tick();
      timer = setInterval(tick, 1000);
      setStatus("Analysing…");
    };
    xhr.onload = () => {
      finish();
      if (xhr.status >= 200 && xhr.status < 300) {
        let result;
        try {
          result = JSON.parse(xhr.responseText);
        } catch (e) {
          setStatus("The server answered, but not with JSON.", true);
          return;
        }
        setStatus("Done.");
        UI.renderResult(resultEl, result);
      } else {
        setStatus(UI.errorMessage(xhr.status, xhr.responseText, "POST " + ENDPOINT), true);
      }
    };
    xhr.onerror = () => { finish(); setStatus("Network error: is the phone on the same Wi-Fi as the laptop, and is the server running?", true); };
    xhr.ontimeout = () => { finish(); setStatus("Timed out waiting for the laptop.", true); };
    xhr.onabort = () => { finish(); setStatus("Upload cancelled."); };
    xhr.send(form);
  }

  function finish() {
    clearInterval(timer);
    timer = 0;
    busy(false);
  }

  document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", applyMode));
  const saved = new URLSearchParams(location.search).get("mode") || UI.loadPref("mode", "finger");
  const radio = document.querySelector('input[name="mode"][value="' + (saved === "face" ? "face" : "finger") + '"]');
  if (radio) radio.checked = true;
  applyMode();

  record.addEventListener("change", () => choose(record));
  pick.addEventListener("change", () => choose(pick));
  send.addEventListener("click", upload);
  cancel.addEventListener("click", () => { if (xhr) xhr.abort(); });
})();
