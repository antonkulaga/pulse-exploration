"""Tests for the phone pages router (pulse_exploration.pages_api) and scripts/make_cert.py.

The app under test includes only the pages router. JavaScript is syntax-checked with `node --check`
and the pure helpers (result rendering, trace building) are exercised under Node; those tests skip
when node is not installed.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # the project is not installed as a package; `uv run pytest` needs this
    sys.path.insert(0, str(ROOT))

from pulse_exploration import pages_api  # noqa: E402

STATIC = pages_api.STATIC
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")

SCRIPT_TAG = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)
SRC_ATTR = re.compile(r"""\bsrc\s*=\s*["']([^"']+)["']""", re.I)
HREF_ATTR = re.compile(r"""<link\b[^>]*\bhref\s*=\s*["']([^"']+)["']""", re.I)


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(pages_api.router)
    return TestClient(app)


def page_scripts(client: TestClient, html: str) -> list[tuple[str, str]]:
    """(name, code) for every inline script and every same-origin script the page loads."""
    out = []
    for i, (attrs, code) in enumerate(SCRIPT_TAG.findall(html)):
        src = SRC_ATTR.search(attrs)
        if src:
            r = client.get(src.group(1))
            assert r.status_code == 200, src.group(1)
            out.append((src.group(1), r.text))
        elif code.strip():
            out.append((f"inline-{i}", code))
    return out


def page_text(client: TestClient, path: str) -> str:
    """The page HTML plus the JavaScript it loads, for endpoint checks."""
    html = client.get(path).text
    return html + "\n".join(code for _, code in page_scripts(client, html))


@pytest.mark.parametrize("path", ["/", "/upload", "/capture"])
def test_pages_are_html(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>" in r.text and 'name="viewport"' in r.text


def test_index_links_both_tiers(client):
    html = client.get("/").text
    assert 'href="/upload"' in html
    assert 'href="/capture"' in html
    assert "tier: video" in html and "tier: capture" in html
    assert "certificate" in html.lower()


def test_upload_page_contract(client):
    html = client.get("/upload").text
    text = page_text(client, "/upload")
    assert 'accept="video/*"' in html
    assert 'capture="environment"' in html
    assert '"user"' in text  # switched by mode in upload.js
    assert 'value="finger"' in html and 'value="face"' in html
    assert "/camera/video" in text
    assert 'append("file"' in text and 'append("mode"' in text
    assert "upload.onprogress" in text and "<progress" in html
    assert "tier: video" in html
    assert "getUserMedia" not in text  # tier video must not need the camera API


def test_capture_page_contract(client):
    html = client.get("/capture").text
    text = page_text(client, "/capture")
    assert "/camera/trace" in text
    for needle in [
        "getUserMedia", "facingMode", "torch", "applyConstraints", "getCapabilities", "exposureMode",
        "whiteBalanceMode", "requestVideoFrameCallback", "requestAnimationFrame", "mediaTime",
        "expectedDisplayTime", "user_agent", "track_settings", "capabilities", "applied_constraints",
        "measured_fps", "isSecureContext",
    ]:
        assert needle in text, needle
    assert 'data-duration-s="45"' in html
    assert "certificate warning" in html.lower() and "expected" in html.lower()
    assert "tier: capture" in html


@pytest.mark.parametrize("path", ["/", "/upload", "/capture"])
def test_page_assets_resolve(client, path):
    html = client.get(path).text
    refs = SRC_ATTR.findall(html) + HREF_ATTR.findall(html)
    assert refs
    for ref in refs:
        if ref.startswith("/static/"):
            r = client.get(ref)
            assert r.status_code == 200, ref
            if ref.endswith(".js"):
                assert "javascript" in r.headers["content-type"]
            if ref.endswith(".css"):
                assert r.headers["content-type"].startswith("text/css")


@pytest.mark.parametrize("path", [
    "/static/../pages_api.py", "/static/..%2Fpages_api.py", "/static/.hidden", "/static/missing.js",
    "/static/%2e%2e", "/static/pages_api.py",
])
def test_static_rejects_bad_paths(client, path):
    assert client.get(path).status_code == 404


@needs_node
@pytest.mark.parametrize("path", ["/", "/upload", "/capture"])
def test_page_javascript_syntax(client, path, tmp_path):
    scripts = page_scripts(client, client.get(path).text)
    assert scripts, f"{path} loads no JavaScript"
    for name, code in scripts:
        f = tmp_path / (re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("_") or "script")
        f = f.with_suffix(".js")
        f.write_text(code)
        r = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{path} {name}: {r.stderr}"


@needs_node
def test_all_static_js_syntax():
    for f in STATIC.glob("*.js"):
        r = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f.name}: {r.stderr}"


NODE_HARNESS = r"""
const assert = require("assert");
const UI = require(process.argv[2]);
const SIG = require(process.argv[3]);

// --- result rendering never throws, whatever subset of keys is present ---
const full = {
  tier: "video", mode: "finger", hr_bpm: 72.4, hr_peaks_bpm: 71.9,
  quality: { snr_db: 6.2, sqi: 0.81, clipped_fraction: 0 },
  warnings: ["auto exposure drift <script>alert(1)</script>"],
  hrv: { rmssd_ms: 35.2, sdnn_ms: 41.0, hrv_reliable: false },
  resp_rate: 14.2,
  deep: { "FacePhys.rlap": { hr: 73.1, sqi: 0.6 }, TSCAN: 70, PhysNet: "failed: no face", empty: null },
  waveform: Array.from({ length: 300 }, (_, i) => Math.sin(i / 5)),
  waveform_fs: 10, extra_field: { nested: { deep: [1, 2, 3] } }, n_frames: 1350,
};
const html = UI.resultHTML(full);
assert(html.includes("72"), "big HR");
assert(html.includes("tier: video"), "tier badge");
assert(html.includes("data-wave"), "waveform canvas");
assert(html.includes("not reliable"), "hrv flag");
assert(html.includes("FacePhys.rlap") && html.includes("73.1"), "deep table");
assert(html.includes("extra field") && html.includes("1350"), "unknown keys shown");
assert(!html.includes("<script>"), "warnings are escaped");

// Shape produced by pulse_exploration.camera.analyze_trace (object warnings, respiration, wrapped deep).
const shaped = {
  analysis_version: "x", tier: "capture", mode: "face", hr_bpm: 72.0, hr_peaks_bpm: 72.0,
  primary_method: "pos", primary_reason: "highest SNR",
  methods: { pos: { hr_fft_bpm: 72.0, hr_peaks_bpm: 72.0, snr_db: 9.1, n_peaks: 54 }, green: { hr_fft_bpm: 70.1 } },
  hrv: null, hrv_reliable: false, hrv_note: "HRV needs >= 60 s of recording.",
  respiration: { rate_brpm: 15.0, reliable: false, estimates: { baseline: { rate_brpm: 15.0, peak_share: 0.99 } } },
  quality: { snr_db: 29.7, template_corr: 0.999, label: "good" },
  warnings: [{ code: "hrv_needs_60s", severity: "info", message: "HRV needs at least 60 s & is not reported." },
             { code: "low_quality", severity: "warning", message: "Signal quality is poor" }],
  waveform: { fs: 25, t_s: [0, 0.04, 0.08, 0.12, 0.16], values: [0, 1, 0, -1, 0], peaks_t_s: [0.041, 0.5], signal: "pos" },
  deep: { package: "open-rppg", available: true, version: "0.1.1", total_s: 12.5,
          models: { "FacePhys.rlap": { hr_bpm: 71.3, sqi: 0.72, snr_db: 4.1, load_s: 3.2, hrv: {}, hrv_reliable: false },
                    PhysNet: { error: "model produced no signal (no face / too short)" } } },
  ibi_ms: Array.from({ length: 50 }, () => 833),
};
const sh = UI.resultHTML(shaped);
assert(sh.includes("HRV needs at least 60 s &amp; is not reported."), "object warning message");
assert(sh.includes("quality: good"), "quality label badge");
assert(sh.includes("HRV needs &gt;= 60 s"), "hrv note");
assert(sh.includes("<b>15.0</b> breaths/min") || sh.includes("<b>15</b> breaths/min"), "respiration rate");
assert(sh.includes("71.3") && sh.includes("SQI 0.720") && sh.includes("model produced no signal"), "deep models");
assert(sh.includes("Per-method estimates") && sh.includes("signal: pos"), "methods and primary");
assert(!sh.includes(">models<"), "deep wrapper is unpacked");
const swave = UI.parseWaveform(shaped);
assert.deepStrictEqual(swave.y, [0, 1, 0, -1, 0]);
assert.deepStrictEqual(swave.peaks, [1], "peaks_t_s mapped to indices, out of range dropped");
assert.strictEqual(swave.label, "pos");
assert(UI.resultHTML({ deep: { available: false, reason: "open-rppg is not installed" } }).includes("not installed"));
assert(UI.resultHTML({ deep: { available: true, skipped: true } }).includes("Skipped"));

const odd = [
  {}, null, [], "text", 5, { tier: "capture" },
  { hr_bpm: null, hr_peaks_bpm: "n/a", warnings: "one string", hrv: null, deep: [], waveform: [] },
  { hr_bpm: "68", warnings: [], quality: 0.4, hrv: 12, resp_rate: { bpm: 15, reliable: false }, deep: "skipped" },
  { hr_bpm: NaN, waveform: { y: [1, null, 3, 4], t: [0, 1, 2, 3], peaks: [2, 99, -1] } },
  { waveform: [[0, 1], [0.1, 2], [0.2, 1.5]], deep: [{ model: "ME-flow", hr_bpm: 75 }, 3] },
  { waveform: { values: [1] } }, { waveform: "nope" }, { hrv: { hrv_reliable: true, rmssd_ms: 40 } },
];
for (const r of odd) {
  const out = UI.resultHTML(r);
  assert(typeof out === "string" && out.length > 0);
}
assert(UI.resultHTML({}).includes("tier missing"));
assert(UI.resultHTML({ hr_bpm: "68" }).includes(">68<"));
assert(UI.resultHTML({ tier: "capture" }).includes("No heart rate"));

// --- waveform parsing ---
assert.deepStrictEqual(UI.parseWaveform({ waveform: [1, 2, 3] }).y, [1, 2, 3]);
assert.strictEqual(UI.parseWaveform({ waveform: [1] }), null);
assert.strictEqual(UI.parseWaveform({}), null);
const pw = UI.parseWaveform({ waveform: { y: [1, 2, 3, 4], fs: 25, peaks: [1, 9] } });
assert.strictEqual(pw.fs, 25);
assert.deepStrictEqual(pw.peaks, [1]);
assert.deepStrictEqual(UI.parseWaveform({ waveform: [[0, 5], [1, 6]] }).t, [0, 1]);

// --- error messages ---
assert(UI.errorMessage(404, '{"detail":"Not Found"}', "POST /camera/trace").includes("does not offer"));
assert(UI.errorMessage(422, '{"detail":[{"msg":"field required"}]}', "x").includes("field required"));
assert(UI.errorMessage(500, "<html>oops</html>", "x").includes("500"));

// --- ROI statistics ---
const px = new Uint8ClampedArray([200, 40, 20, 255, 100, 60, 0, 255, 255, 10, 10, 255, 0, 0, 0, 255]);
const s = SIG.roiStats(px);
assert.strictEqual(s.n, 4);
assert(Math.abs(s.r - 138.75) < 1e-9 && Math.abs(s.g - 27.5) < 1e-9 && Math.abs(s.b - 7.5) < 1e-9);
assert.strictEqual(s.sat, 0.25);
assert.strictEqual(SIG.roiStats(new Uint8ClampedArray(0)), null);

// --- timestamps and trace assembly ---
const frames = [];
for (let i = 0; i < 300; i++) {
  frames.push({
    mediaTime: 5000 + i * 33.333, captureTime: 900 + i * 33.3 + (i % 3), expectedDisplayTime: 1000 + i * 33.4,
    now: 1001 + i * 33.4, presentedFrames: i + (i > 100 ? 2 : 0), r: 200 + Math.sin(i), g: 40.12345, b: 20, sat: i === 7 ? 0.5 : 0,
  });
}
const tr = SIG.buildTrace(frames, "finger", { user_agent: "test" });
assert.deepStrictEqual(Object.keys(tr).sort(), ["b", "g", "meta", "mode", "r", "t_ms"]);
assert.strictEqual(tr.mode, "finger");
assert.strictEqual(tr.t_ms.length, 300);
assert.strictEqual(tr.r.length, 300);
assert.strictEqual(tr.t_ms[0], 0);
assert.strictEqual(tr.g[0], 40.123);
assert.strictEqual(tr.meta.timestamp_source, "mediaTime");
assert.strictEqual(tr.meta.user_agent, "test");
assert(Math.abs(tr.meta.measured_fps.median_fps - 30) < 0.1, "fps " + tr.meta.measured_fps.median_fps);
assert.strictEqual(tr.meta.frames_missed, 2);
assert.strictEqual(tr.meta.saturation.max_fraction, 0.5);
assert(tr.meta.timestamps_alt_ms.captureTime.length === 300);

// mediaTime missing or stuck -> next source
const noMedia = frames.map((f) => Object.assign({}, f, { mediaTime: null }));
assert.strictEqual(SIG.chooseTimestamps(noMedia).source, "captureTime");
const stuck = frames.map((f, i) => Object.assign({}, f, { mediaTime: 0, captureTime: undefined }));
assert.strictEqual(SIG.chooseTimestamps(stuck).source, "expectedDisplayTime");
assert.strictEqual(SIG.framesMissed([{ presentedFrames: null }]), null);

// detrend removes a constant offset
const d = SIG.detrend([5, 5, 5, 5, 5], 3);
assert(d.every((x) => Math.abs(x) < 1e-12));
console.log("OK");
"""


@needs_node
def test_js_helpers_under_node(tmp_path):
    harness = tmp_path / "harness.js"
    harness.write_text(NODE_HARNESS)
    r = subprocess.run(
        [NODE, str(harness), str(STATIC / "common.js"), str(STATIC / "signal.js")],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert r.stdout.strip().endswith("OK")


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not installed")
def test_make_cert_writes_san(tmp_path):
    out = tmp_path / "certs"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_cert.py"), "--out", str(out), "--ip", "192.168.1.177"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "cert.pem").exists() and (out / "key.pem").exists()
    assert oct((out / "key.pem").stat().st_mode & 0o777) == "0o600"
    san = subprocess.run(
        ["openssl", "x509", "-in", str(out / "cert.pem"), "-noout", "-ext", "subjectAltName,basicConstraints,extendedKeyUsage"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "IP Address:192.168.1.177" in san and "DNS:localhost" in san and "IP Address:127.0.0.1" in san
    assert "CA:FALSE" in san and "TLS Web Server Authentication" in san
    assert "--port 8766" in r.stdout and "--ssl-keyfile" in r.stdout and "--ssl-certfile" in r.stdout
    # A second run keeps the existing certificate.
    before = (out / "cert.pem").read_bytes()
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_cert.py"), "--out", str(out)],
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0 and (out / "cert.pem").read_bytes() == before
    assert "Kept existing" in r2.stdout
