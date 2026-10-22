package io.github.antonkulaga.pulsephone

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Collects frames for one recording. The duration is counted on the sensor clock, so a run holds exactly
 * [seconds] of frames however late the analyser or the UI is. Frames captured before [minTsNs] (the
 * sensor time of the first result showing the AE/AWB locks) are skipped: the analyser runs a frame or two
 * behind the capture results, so the first frames delivered after the lock was confirmed can predate it.
 */
class TraceRecorder(private val seconds: Int, private val minTsNs: Long = 0L) {
    private val frames = ArrayList<Frame>(seconds * 64)
    private var done = false
    @Volatile var elapsedNs = 0L
        private set
    @Volatile var count = 0
        private set

    /** Adds [f]; returns true exactly once, when the requested duration is covered. */
    @Synchronized
    fun offer(f: Frame): Boolean {
        if (done || f.tNs < minTsNs) return false
        if (frames.isNotEmpty() && f.tNs <= frames.last().tNs) return false
        frames += f
        count = frames.size
        elapsedNs = f.tNs - frames.first().tNs
        if (elapsedNs >= seconds * 1_000_000_000L) {
            done = true
            return true
        }
        return false
    }

    @Synchronized
    fun stop(): List<Frame> {
        done = true
        return ArrayList(frames)
    }
}

/** Frame rate over the last [window] sensor timestamps, for the live display. */
class FpsMeter(window: Int = 30) {
    private val ts = LongArray(window + 1)
    private var n = 0
    private var head = 0

    @Synchronized
    fun add(tNs: Long) {
        ts[head] = tNs
        head = (head + 1) % ts.size
        if (n < ts.size) n++
    }

    @Synchronized
    fun fps(): Double {
        if (n < 2) return 0.0
        val newest = ts[(head - 1 + ts.size) % ts.size]
        val oldest = ts[(head - n + ts.size) % ts.size]
        return if (newest > oldest) (n - 1) * 1e9 / (newest - oldest) else 0.0
    }
}

/** Everything about a run that is not the frames themselves. */
class TraceContext(
    val config: RecordingConfig,
    val app: JSONObject,
    val device: JSONObject,
    val camera: JSONObject,
    val applied: JSONObject,
    val echo: JSONObject,
    val geometry: FrameGeometry?,
    val roiInCrop: Boolean,
    val stoppedEarly: Boolean,
    val recordedUtc: String,
)

/**
 * Builds the body of `POST /camera/trace` and the per-frame CSV kept next to it.
 *
 * t_ms is sensor time relative to the first frame (meta.timing.t0_sensor_ns is the origin); r/g/b are
 * ROI means in 0..255. Per-frame clipping and capture-result values stay in frames.csv, so meta (the
 * server caps it at 1 MB) stays a few KB and only carries summaries, including the clipped_fraction_* keys it reads.
 */
object TraceJson {
    class Built(val json: JSONObject, val csv: String, val summary: String)

    fun build(frames: List<Frame>, samples: List<CaptureLog.Sample>, ctx: TraceContext): Built {
        val t0 = frames.first().tNs
        val tLast = frames.last().tNs
        val tMs = frames.map { (it.tNs - t0) / 1e6 }
        val dts = tMs.zipWithNext { a, b -> b - a }
        val med = median(dts) ?: 0.0
        val spanS = (tLast - t0) / 1e9
        val fps = if (spanS > 0) (frames.size - 1) / spanS else 0.0
        val gaps = if (med > 0) dts.count { it > 1.5 * med } else 0
        val missing = if (med > 0) dts.sumOf { max(0, (it / med).roundToInt() - 1) } else 0
        val inRun = samples.filter { it.tNs in t0..tLast }
        val byTs = samples.associateBy { it.tNs }
        val matched = frames.count { it.tNs in byTs }

        val timing = JSONObject().apply {
            put("t0_sensor_ns", t0)
            put("timestamp_source", ctx.camera.optString("timestamp_source", "unknown"))
            put("frames", frames.size)
            put("duration_s", num(spanS, 3))
            put("measured_fps", num(fps, 3))
            put("frame_interval_ms_median", num(med, 3))
            put("frame_gaps", gaps)
            put("frames_missing_est", missing)
            // Frames the camera produced in the window vs. frames the analyser got (KEEP_ONLY_LATEST drops).
            put("frames_captured", inRun.size)
            put("frames_dropped_by_analysis", max(0, inRun.size - frames.size))
            put("frames_matched_to_capture_result", matched)
        }
        val roi = JSONObject().apply {
            put("relative_to", if (ctx.roiInCrop) "visible preview (crop rect)" else "full frame")
            val f = if (ctx.config.mode == Mode.FINGER) Roi.FINGER else Roi.FACE
            put("fraction_upright", JSONArray().put(num(f.left.toDouble(), 3)).put(num(f.top.toDouble(), 3))
                .put(num(f.right.toDouble(), 3)).put(num(f.bottom.toDouble(), 3)))
            ctx.geometry?.let { g ->
                put("image", JSONArray().put(g.width).put(g.height))
                put("rotation_degrees", g.rotationDegrees)
                put("crop", JSONArray().put(g.crop.left).put(g.crop.top).put(g.crop.right).put(g.crop.bottom))
                put("buffer", JSONArray().put(g.roi.left).put(g.roi.top).put(g.roi.right).put(g.roi.bottom))
                put("pixel_format", g.format)
                put("sampling_step", g.step)
            }
            put("clip_threshold", FrameAnalyzer.CLIP)
        }
        val meta = JSONObject().apply {
            put("source", "pulsephone")
            put("mode", ctx.config.mode.wire)
            put("duration_s_requested", ctx.config.seconds)
            put("duration_s_recorded", num(spanS, 3))
            put("stopped_early", ctx.stoppedEarly)
            put("recorded_utc", ctx.recordedUtc)
            put("app", ctx.app)
            put("device", ctx.device)
            put("camera", ctx.camera)
            put("applied", ctx.applied)
            put("capture_result", CaptureLog.stats(inRun).put("echo", ctx.echo))
            put("timing", timing)
            put("roi", roi)
            put("clipped_fraction_any", num(frames.map { it.clipAny }.average(), 5))
            put("clipped_fraction_r", num(frames.map { it.clipR }.average(), 5))
            put("clipped_fraction_g", num(frames.map { it.clipG }.average(), 5))
            put("clipped_fraction_b", num(frames.map { it.clipB }.average(), 5))
        }
        val json = JSONObject().apply {
            put("mode", ctx.config.mode.wire)
            put("t_ms", JSONArray().apply { tMs.forEach { put(num(it, 3)) } })
            put("r", JSONArray().apply { frames.forEach { put(num(it.r)) } })
            put("g", JSONArray().apply { frames.forEach { put(num(it.g)) } })
            put("b", JSONArray().apply { frames.forEach { put(num(it.b)) } })
            put("meta", meta)
        }

        val csv = StringBuilder("t_ms,sensor_ns,r,g,b,clip_r,clip_any,exposure_ns,iso,frame_duration_ns,ae_state,awb_state\n")
        frames.forEachIndexed { i, f ->
            val s = byTs[f.tNs]
            csv.append(String.format(Locale.US, "%.3f,%d,%.4f,%.4f,%.4f,%.5f,%.5f,", tMs[i], f.tNs, f.r, f.g, f.b, f.clipR, f.clipAny))
            csv.append(s?.exposureNs ?: "").append(',').append(s?.iso ?: "").append(',').append(s?.frameDurationNs ?: "")
                .append(',').append(s?.aeState?.let(Names::aeState) ?: "").append(',').append(s?.awbState?.let(Names::awbState) ?: "")
                .append('\n')
        }
        val summary = String.format(
            Locale.US, "%d frames, %.1f s, %.2f fps, %d gaps, clip_r %.3f, %d/%d capture results matched",
            frames.size, spanS, fps, gaps, frames.map { it.clipR }.average(), matched, frames.size,
        )
        return Built(json, csv.toString(), summary)
    }
}
