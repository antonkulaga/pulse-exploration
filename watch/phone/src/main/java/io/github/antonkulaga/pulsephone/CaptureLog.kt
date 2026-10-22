package io.github.antonkulaga.pulsephone

import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

/**
 * What the camera says it actually did, frame by frame. Fed by a Camera2Interop session capture
 * callback; during a recording every result is kept, so exposure, ISO and AE state can be joined to the
 * analysed frames by sensor timestamp (an Image's timestamp equals its result's SENSOR_TIMESTAMP).
 */
class CaptureLog {
    class Sample(
        val tNs: Long,
        val exposureNs: Long?,
        val iso: Int?,
        val frameDurationNs: Long?,
        val aeState: Int?,
        val awbState: Int?,
        val aeLock: Boolean?,
        val awbLock: Boolean?,
        val flashState: Int?,
    )

    @Volatile var latest: Sample? = null
        private set
    @Volatile private var latestResult: TotalCaptureResult? = null
    private val window = ArrayList<Sample>()
    /** The last few results before [begin]: the first recorded frame's result often arrives just before it. */
    private val recent = ArrayDeque<Sample>()
    private var collecting = false
    private var waiter: ((Sample) -> Unit)? = null
    private val main = Handler(Looper.getMainLooper())

    val callback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            onResult(result)
        }
    }

    private fun onResult(r: TotalCaptureResult) {
        val s = Sample(
            tNs = r.get(CaptureResult.SENSOR_TIMESTAMP) ?: return,
            exposureNs = r.get(CaptureResult.SENSOR_EXPOSURE_TIME),
            iso = r.get(CaptureResult.SENSOR_SENSITIVITY),
            frameDurationNs = r.get(CaptureResult.SENSOR_FRAME_DURATION),
            aeState = r.get(CaptureResult.CONTROL_AE_STATE),
            awbState = r.get(CaptureResult.CONTROL_AWB_STATE),
            aeLock = r.get(CaptureResult.CONTROL_AE_LOCK),
            awbLock = r.get(CaptureResult.CONTROL_AWB_LOCK),
            flashState = r.get(CaptureResult.FLASH_STATE),
        )
        latest = s
        latestResult = r
        synchronized(this) {
            if (collecting) {
                window += s
            } else {
                recent.addLast(s)
                if (recent.size > RECENT) recent.removeFirst()
            }
            waiter?.invoke(s)
        }
    }

    @Synchronized
    fun begin() {
        window.clear()
        window.addAll(recent) // trimmed to the frames' time span when the trace is built
        recent.clear()
        collecting = true
    }

    @Synchronized
    fun end(): List<Sample> {
        collecting = false
        return ArrayList(window)
    }

    /**
     * Calls [done] on the main thread once a result reports the requested locks (the request echo or the
     * AE/AWB state LOCKED), or with `false` after [timeoutMs]. [lockedAtNs] is the SENSOR_TIMESTAMP of the
     * first result that showed the locks (0 when not confirmed): frames older than it were captured unlocked.
     */
    fun awaitLocks(
        ae: Boolean, awb: Boolean, timeoutMs: Long,
        done: (confirmed: Boolean, latencyMs: Long, lockedAtNs: Long) -> Unit,
    ) {
        val start = SystemClock.elapsedRealtime()
        val fired = AtomicBoolean(false)
        var mine: ((Sample) -> Unit)? = null
        val finish = { ok: Boolean, atNs: Long ->
            if (fired.compareAndSet(false, true)) {
                // Only clear our own waiter: a stale timeout must not cancel a newer run's wait.
                synchronized(this) { if (waiter === mine) waiter = null }
                main.post { done(ok, SystemClock.elapsedRealtime() - start, atNs) }
            }
        }
        if (!ae && !awb) return finish(true, 0L)
        val w = { s: Sample ->
            val aeOk = !ae || s.aeLock == true || s.aeState == CaptureResult.CONTROL_AE_STATE_LOCKED
            val awbOk = !awb || s.awbLock == true || s.awbState == CaptureResult.CONTROL_AWB_STATE_LOCKED
            if (aeOk && awbOk) finish(true, s.tNs)
        }
        mine = w
        synchronized(this) { waiter = w }
        main.postDelayed({ finish(false, 0L) }, timeoutMs)
    }

    /** The request fields the HAL echoed back in the latest result: what was really applied. */
    fun echo(): JSONObject {
        val r = latestResult ?: return JSONObject()
        return JSONObject().apply {
            put("ae_mode", Names.aeMode(r.get(CaptureResult.CONTROL_AE_MODE)))
            put("af_mode", Names.afMode(r.get(CaptureResult.CONTROL_AF_MODE)))
            put("ae_lock", r.get(CaptureResult.CONTROL_AE_LOCK) ?: JSONObject.NULL)
            put("awb_lock", r.get(CaptureResult.CONTROL_AWB_LOCK) ?: JSONObject.NULL)
            put("fps_range", jsonRange(r.get(CaptureResult.CONTROL_AE_TARGET_FPS_RANGE)))
            put("noise_reduction", Names.noiseReduction(r.get(CaptureResult.NOISE_REDUCTION_MODE)))
            put("edge", Names.edge(r.get(CaptureResult.EDGE_MODE)))
            put("video_stabilization", Names.videoStabilization(r.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE)))
            put("optical_stabilization", Names.opticalStabilization(r.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE)))
            put("flash_mode", Names.flashMode(r.get(CaptureResult.FLASH_MODE)))
            put("post_raw_sensitivity_boost", r.get(CaptureResult.CONTROL_POST_RAW_SENSITIVITY_BOOST) ?: JSONObject.NULL)
            put("awb_gains_rggb", r.get(CaptureResult.COLOR_CORRECTION_GAINS)?.let {
                JSONArray().put(num(it.red.toDouble(), 3)).put(num(it.greenEven.toDouble(), 3))
                    .put(num(it.greenOdd.toDouble(), 3)).put(num(it.blue.toDouble(), 3))
            } ?: JSONObject.NULL)
        }
    }

    companion object {
        private const val RECENT = 8

        /** Summary of the results received during a recording, for meta.capture_result. */
        fun stats(samples: List<Sample>): JSONObject = JSONObject().apply {
            put("n", samples.size)
            put("exposure_ns", spread(samples.mapNotNull { it.exposureNs?.toDouble() }))
            put("iso", spread(samples.mapNotNull { it.iso?.toDouble() }))
            put("frame_duration_ns", spread(samples.mapNotNull { it.frameDurationNs?.toDouble() }))
            val fd = median(samples.mapNotNull { it.frameDurationNs?.toDouble() })
            put("fps_from_frame_duration", num(fd?.let { 1e9 / it }, 2))
            put("ae_state", counts(samples.map { Names.aeState(it.aeState) }))
            put("awb_state", counts(samples.map { Names.awbState(it.awbState) }))
            put("flash_state", counts(samples.map { Names.flashState(it.flashState) }))
            val n = samples.size.coerceAtLeast(1).toDouble()
            put("ae_locked_fraction", num(samples.count { it.aeState == CaptureResult.CONTROL_AE_STATE_LOCKED } / n, 3))
            put("awb_locked_fraction", num(samples.count { it.awbState == CaptureResult.CONTROL_AWB_STATE_LOCKED } / n, 3))
        }

        private fun spread(v: List<Double>): JSONObject = JSONObject().apply {
            put("min", num(v.minOrNull(), 1))
            put("median", num(median(v), 1))
            put("max", num(v.maxOrNull(), 1))
            put("distinct", v.distinct().size)
        }

        private fun counts(v: List<String>): JSONObject =
            JSONObject().apply { v.groupingBy { it }.eachCount().forEach { (k, c) -> put(k, c) } }
    }
}

fun median(v: List<Double>): Double? {
    if (v.isEmpty()) return null
    val s = v.sorted()
    return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
}
