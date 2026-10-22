package io.github.antonkulaga.pulsephone

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.CameraState
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import org.json.JSONObject
import java.util.concurrent.Executor

/**
 * The CameraX session behind a recording: a Preview for aiming and an RGBA ImageAnalysis for the colour
 * means, bound together under the preview's ViewPort so the analysis crop rect is what the user sees.
 *
 * Two kinds of Camera2 control, kept on separate keys on purpose:
 * - Static options (AE target fps range, stabilisation, noise reduction, edge, AF off for the finger,
 *   manual exposure) go through Camera2Interop on both use cases with identical values. In CameraX 1.6
 *   these are ALWAYS_OVERRIDE, so they win over CameraX's own defaults, and AE/AF modes found in them are
 *   handed on to CameraPipe's 3A state.
 * - AE/AWB lock changes at runtime, so it goes through Camera2CameraControl. Interop never sets those
 *   keys: two ALWAYS_OVERRIDE sources for one key would silently replace each other.
 */
@OptIn(ExperimentalCamera2Interop::class)
class PpgCamera(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val previewView: PreviewView,
    private val analyzer: ImageAnalysis.Analyzer,
    private val analysisExecutor: Executor,
) {
    val captures = CaptureLog()
    var camera: Camera? = null
        private set
    /** Characteristics of the bound camera (meta.camera). */
    var report = JSONObject()
        private set
    /** What this app asked the camera for (meta.applied); lock and torch entries are added later. */
    var applied = JSONObject()
        private set
    var manualSupported = false
        private set
    var exposureRangeNs: Range<Long>? = null
        private set
    var isoRange: Range<Int>? = null
        private set
    private var aeLockAvailable = false
    private var awbLockAvailable = false
    private var boundConfig: RecordingConfig? = null
    private var manualApplied = false
    private var torchOn = false
    private var generation = 0

    /**
     * Main thread: the bound camera left OPEN after having been open (another client took it, a HAL error,
     * or the app stopping). CameraX reopens it by itself but drops the torch on the way, so a run in
     * progress cannot simply continue.
     */
    var onInterrupted: ((CameraState.Type) -> Unit)? = null

    fun isBound(config: RecordingConfig) = camera != null && boundConfig == config

    /** (Re)binds for [config]; [done] runs on the main thread with null or an error message. */
    fun open(config: RecordingConfig, done: (String?) -> Unit) {
        val gen = ++generation
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (gen != generation) return@addListener
            val provider = try {
                future.get()
            } catch (e: Exception) {
                return@addListener done("camera provider: $e")
            }
            // post: the ViewPort needs the PreviewView laid out.
            previewView.post { if (gen == generation) done(bind(provider, config)) }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bind(provider: ProcessCameraProvider, config: RecordingConfig): String? {
        val selector = if (config.mode == Mode.FINGER) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
        val info = try {
            provider.getCameraInfo(selector)
        } catch (e: Exception) {
            return "no ${if (config.mode == Mode.FINGER) "rear" else "front"} camera: ${e.message}"
        }
        report = describeCamera(context, info)
        val plan = plan(Camera2CameraInfo.from(info), info, config)
        camera?.cameraInfo?.cameraState?.removeObservers(owner)
        provider.unbindAll()
        camera = null
        boundConfig = null
        return try {
            val p = Preview.Builder()
                .setResolutionSelector(resolution(Size(960, 720)))
                .apply { Camera2Interop.Extender(this).applyAll(plan.options) }
                .build()
            val a = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setResolutionSelector(resolution(Size(640, 480)))
                .apply {
                    // Only here: on both use cases every result would arrive twice.
                    Camera2Interop.Extender(this).applyAll(plan.options).setSessionCaptureCallback(captures.callback)
                }
                .build()
            a.setAnalyzer(analysisExecutor, analyzer)
            val viewPort = previewView.viewPort
            val group = UseCaseGroup.Builder().apply {
                viewPort?.let(::setViewPort)
                addUseCase(p)
                addUseCase(a)
            }.build()
            val cam = provider.bindToLifecycle(owner, selector, group)
            p.setSurfaceProvider(previewView.surfaceProvider)
            var lastType: CameraState.Type? = null
            cam.cameraInfo.cameraState.observe(owner) { state ->
                state.error?.let { Log.w(TAG, "camera state ${state.type}: error ${it.code} ${it.cause ?: ""}") }
                val wasOpen = lastType == CameraState.Type.OPEN
                lastType = state.type
                if (wasOpen && state.type != CameraState.Type.OPEN) {
                    Log.w(TAG, "camera left OPEN: ${state.type}")
                    // CameraX turns the torch off while the camera is closed and does not restore it.
                    torchOn = false
                    if (manualApplied) boundConfig = null // same AE-mode quirk as switching the torch off
                    onInterrupted?.invoke(state.type)
                }
            }
            camera = cam
            boundConfig = config
            torchOn = false
            manualApplied = plan.applied.optString("exposure") == "manual"
            applied = plan.applied.apply {
                put("viewport", viewPort != null)
                put("analysis_format", "RGBA_8888")
                put("analysis_resolution", a.resolutionInfo?.resolution?.toString() ?: JSONObject.NULL)
                put("preview_resolution", p.resolutionInfo?.resolution?.toString() ?: JSONObject.NULL)
            }
            Log.i(TAG, "bound camera ${report.optString("camera_id")} ${config.mode.wire}: $applied")
            null
        } catch (e: Exception) {
            provider.unbindAll()
            if (config.try60) {
                Log.w(TAG, "bind with 60 fps failed ($e), retrying at 30 fps")
                bind(provider, config.copy(try60 = false))?.let { return it }
                boundConfig = config // so the same request does not rebind every time
                applied.put("fps_60_fallback", e.toString())
                null
            } else {
                "bind failed: $e"
            }
        }
    }

    private class Plan(val options: List<Pair<CaptureRequest.Key<*>, Any>>, val applied: JSONObject)

    /** Picks the static Camera2 options for [config] from what the camera lists. Best-effort throughout. */
    private fun plan(c2: Camera2CameraInfo, info: CameraInfo, config: RecordingConfig): Plan {
        fun <T> get(key: CameraCharacteristics.Key<T>): T? = try {
            c2.getCameraCharacteristic(key)
        } catch (_: Exception) {
            null
        }
        val options = mutableListOf<Pair<CaptureRequest.Key<*>, Any>>()
        val applied = JSONObject()

        val ranges = get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()
        val fps = chooseFps(ranges, config.try60)
        applied.put("fps_60_requested", config.try60)
        applied.put("fps_range", jsonRange(fps))
        if (fps != null) options += CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE to fps

        /** Sets [key] to the first of [preferred] that the camera lists, and records which (or "not listed"). */
        fun firstListed(name: String, key: CaptureRequest.Key<Int>, listed: IntArray?, preferred: List<Int>, label: (Int?) -> String) {
            val mode = preferred.firstOrNull { listed?.contains(it) == true }
            if (mode != null) options += key to mode
            applied.put(name, if (mode != null) label(mode) else "not listed")
        }
        firstListed("video_stabilization", CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
            get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES),
            listOf(CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF), Names::videoStabilization)
        firstListed("optical_stabilization", CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
            get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION),
            listOf(CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF), Names::opticalStabilization)
        // Temporal noise reduction averages across frames and would smear the pulse; MINIMAL is the fallback.
        firstListed("noise_reduction", CaptureRequest.NOISE_REDUCTION_MODE,
            get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES),
            listOf(CameraMetadata.NOISE_REDUCTION_MODE_OFF, CameraMetadata.NOISE_REDUCTION_MODE_MINIMAL), Names::noiseReduction)
        firstListed("edge", CaptureRequest.EDGE_MODE, get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES),
            listOf(CameraMetadata.EDGE_MODE_OFF), Names::edge)
        if (config.mode == Mode.FINGER) {
            // Nothing to focus on under a fingertip; continuous AF would hunt and move the lens all run.
            firstListed("af_mode", CaptureRequest.CONTROL_AF_MODE, get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES),
                listOf(CameraMetadata.CONTROL_AF_MODE_OFF), Names::afMode)
        } else {
            applied.put("af_mode", "camerax default")
        }

        val caps = get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        manualSupported = caps?.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true
        exposureRangeNs = get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        isoRange = get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        aeLockAvailable = get(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE) == true
        awbLockAvailable = get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE) == true

        val manual = config.manual
        if (manual != null && manualSupported) {
            // With AE off the fps range no longer applies: the frame duration has to be set as well, and the
            // exposure has to fit inside it.
            val frameNs = 1_000_000_000L / (fps?.upper ?: 30)
            val exposure = exposureRangeNs?.clamp(manual.exposureNs) ?: manual.exposureNs
            val iso = isoRange?.clamp(manual.iso) ?: manual.iso
            val fitted = exposure.coerceAtMost(frameNs)
            options += CaptureRequest.CONTROL_AE_MODE to CameraMetadata.CONTROL_AE_MODE_OFF
            options += CaptureRequest.SENSOR_EXPOSURE_TIME to fitted
            options += CaptureRequest.SENSOR_SENSITIVITY to iso
            options += CaptureRequest.SENSOR_FRAME_DURATION to frameNs
            applied.put("exposure", "manual")
            applied.put("exposure_ns", fitted)
            applied.put("iso", iso)
            applied.put("frame_duration_ns", frameNs)
        } else {
            applied.put("exposure", if (manual != null) "auto (manual asked, MANUAL_SENSOR not listed)" else "auto, locked after settling")
        }
        applied.put("flash_unit", info.hasFlashUnit())
        return Plan(options, applied)
    }

    /** Torch for the fingertip. Returns what happened, for meta.applied.torch. */
    fun setTorch(on: Boolean): String {
        val cam = camera ?: return "no camera"
        if (!cam.cameraInfo.hasFlashUnit()) return "no flash unit"
        if (on == torchOn) return if (on) "on" else "off"
        cam.cameraControl.enableTorch(on)
        torchOn = on
        // CameraX 1.6 switches the torch off with its own AE mode (ON), which replaces the interop
        // AE_MODE_OFF in CameraPipe's 3A state for the rest of the session (switching it on keeps OFF).
        // So after a manual-exposure run the next one has to rebind.
        if (!on && manualApplied) boundConfig = null
        return if (on) "on" else "off"
    }

    /**
     * Locks AE (unless [lockAe] is false, e.g. with manual exposure) and AWB where the camera allows it.
     * [done] runs on the main thread once a capture result confirms the locks, or after a timeout.
     */
    fun lock(lockAe: Boolean, done: (confirmed: Boolean, latencyMs: Long, lockedAtNs: Long) -> Unit) {
        val cam = camera ?: return done(false, 0, 0)
        val ae = lockAe && aeLockAvailable
        val awb = awbLockAvailable
        applied.put("ae_lock", if (ae) "requested" else if (lockAe) "not available" else "not needed (manual)")
        applied.put("awb_lock", if (awb) "requested" else "not available")
        captures.awaitLocks(ae, awb, LOCK_TIMEOUT_MS, done)
        val options = CaptureRequestOptions.Builder().apply {
            if (ae) setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, true)
            if (awb) setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, true)
        }.build()
        Camera2CameraControl.from(cam.cameraControl).setCaptureRequestOptions(options)
    }

    /** Lets AE/AWB run again (explicit false rather than clearing, so no template default can keep a lock). */
    fun unlock() {
        val cam = camera ?: return
        val options = CaptureRequestOptions.Builder().apply {
            if (aeLockAvailable) setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, false)
            if (awbLockAvailable) setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, false)
        }.build()
        Camera2CameraControl.from(cam.cameraControl).setCaptureRequestOptions(options)
    }

    private fun resolution(size: Size) = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(ResolutionStrategy(size, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
        .build()

    private fun <T> Camera2Interop.Extender<T>.applyAll(options: List<Pair<CaptureRequest.Key<*>, Any>>): Camera2Interop.Extender<T> {
        for ((key, value) in options) {
            @Suppress("UNCHECKED_CAST")
            setCaptureRequestOption(key as CaptureRequest.Key<Any>, value)
        }
        return this
    }

    companion object {
        const val TAG = "PulsePhone"
        private const val LOCK_TIMEOUT_MS = 1500L

        /**
         * (60,60) when asked for and listed, else (30,30), else the range topping out at 30 with the highest
         * floor (e.g. [24,30]). A fixed range keeps AE from trading frame rate for brightness mid-run.
         */
        fun chooseFps(ranges: Array<out Range<Int>>, want60: Boolean): Range<Int>? {
            fun fixed(f: Int) = ranges.firstOrNull { it.lower == f && it.upper == f }
            if (want60) fixed(60)?.let { return it }
            return fixed(30) ?: ranges.filter { it.upper == 30 }.maxByOrNull { it.lower }
        }
    }
}
