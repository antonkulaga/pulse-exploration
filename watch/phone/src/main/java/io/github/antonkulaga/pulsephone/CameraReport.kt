package io.github.antonkulaga.pulsephone

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.util.Range
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.roundToLong

/** Camera2 enum values as the names the docs use, so meta reads without a lookup table. */
object Names {
    private val capabilities = listOf(
        "BACKWARD_COMPATIBLE", "MANUAL_SENSOR", "MANUAL_POST_PROCESSING", "RAW", "PRIVATE_REPROCESSING",
        "READ_SENSOR_SETTINGS", "BURST_CAPTURE", "YUV_REPROCESSING", "DEPTH_OUTPUT", "CONSTRAINED_HIGH_SPEED_VIDEO",
        "MOTION_TRACKING", "LOGICAL_MULTI_CAMERA", "MONOCHROME", "SECURE_IMAGE_DATA", "SYSTEM_CAMERA",
        "OFFLINE_PROCESSING", "ULTRA_HIGH_RESOLUTION_SENSOR", "REMOSAIC_REPROCESSING", "DYNAMIC_RANGE_TEN_BIT",
        "STREAM_USE_CASE", "COLOR_SPACE_PROFILES",
    )
    private val hardwareLevels = listOf("LIMITED", "FULL", "LEGACY", "LEVEL_3", "EXTERNAL")
    private val aeStates = listOf("INACTIVE", "SEARCHING", "CONVERGED", "LOCKED", "FLASH_REQUIRED", "PRECAPTURE")
    private val awbStates = listOf("INACTIVE", "SEARCHING", "CONVERGED", "LOCKED")
    private val aeModes = listOf("OFF", "ON", "ON_AUTO_FLASH", "ON_ALWAYS_FLASH", "ON_AUTO_FLASH_REDEYE", "ON_EXTERNAL_FLASH", "ON_LOW_LIGHT_BOOST")
    private val afModes = listOf("OFF", "AUTO", "MACRO", "CONTINUOUS_VIDEO", "CONTINUOUS_PICTURE", "EDOF")
    private val noiseReduction = listOf("OFF", "FAST", "HIGH_QUALITY", "MINIMAL", "ZERO_SHUTTER_LAG")
    private val edge = listOf("OFF", "FAST", "HIGH_QUALITY", "ZERO_SHUTTER_LAG")
    private val videoStab = listOf("OFF", "ON", "PREVIEW_STABILIZATION")
    private val onOff = listOf("OFF", "ON")
    private val flashStates = listOf("UNAVAILABLE", "CHARGING", "READY", "FIRED", "PARTIAL")
    private val flashModes = listOf("OFF", "SINGLE", "TORCH")
    private val timestampSources = listOf("UNKNOWN", "REALTIME")

    private fun List<String>.of(v: Int?): String = if (v == null) "null" else getOrNull(v) ?: v.toString()

    fun capability(v: Int) = capabilities.of(v)
    fun hardwareLevel(v: Int?) = hardwareLevels.of(v)
    fun aeState(v: Int?) = aeStates.of(v)
    fun awbState(v: Int?) = awbStates.of(v)
    fun aeMode(v: Int?) = aeModes.of(v)
    fun afMode(v: Int?) = afModes.of(v)
    fun noiseReduction(v: Int?) = noiseReduction.of(v)
    fun edge(v: Int?) = edge.of(v)
    fun videoStabilization(v: Int?) = videoStab.of(v)
    fun opticalStabilization(v: Int?) = onOff.of(v)
    fun flashState(v: Int?) = flashStates.of(v)
    fun flashMode(v: Int?) = flashModes.of(v)
    fun timestampSource(v: Int?) = timestampSources.of(v)
}

/** JSON helpers: org.json throws on NaN, so every number goes through [num]. */
fun num(v: Double?, digits: Int = 4): Any {
    if (v == null || !v.isFinite()) return JSONObject.NULL
    val scale = Math.pow(10.0, digits.toDouble())
    return (v * scale).roundToLong() / scale
}

fun jsonInts(values: IntArray?, name: (Int) -> String = { it.toString() }): JSONArray =
    JSONArray().apply { values?.forEach { put(name(it)) } }

fun jsonRange(r: Range<*>?): Any = r?.let { JSONArray().put(it.lower).put(it.upper) } ?: JSONObject.NULL

/** Static description of the camera: what it lists, independent of what this app requested. */
@OptIn(ExperimentalCamera2Interop::class)
fun describeCamera(context: Context, info: CameraInfo): JSONObject {
    val c2 = Camera2CameraInfo.from(info)
    fun <T> get(key: CameraCharacteristics.Key<T>): T? = try {
        c2.getCameraCharacteristic(key)
    } catch (_: Exception) {
        null
    }
    val physical = try {
        context.getSystemService(CameraManager::class.java).getCameraCharacteristics(c2.cameraId).physicalCameraIds.toList()
    } catch (_: Exception) {
        emptyList()
    }
    return JSONObject().apply {
        put("camera_id", c2.cameraId)
        put("lens_facing", when (get(CameraCharacteristics.LENS_FACING)) {
            CameraCharacteristics.LENS_FACING_BACK -> "back"
            CameraCharacteristics.LENS_FACING_FRONT -> "front"
            CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
            else -> "unknown"
        })
        put("hardware_level", Names.hardwareLevel(get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)))
        put("capabilities", jsonInts(get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES), Names::capability))
        put("fps_ranges", JSONArray().apply {
            get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.forEach { put(jsonRange(it)) }
        })
        put("timestamp_source", Names.timestampSource(get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)))
        put("exposure_time_range_ns", jsonRange(get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)))
        put("sensitivity_range", jsonRange(get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)))
        put("max_analog_sensitivity", get(CameraCharacteristics.SENSOR_MAX_ANALOG_SENSITIVITY) ?: JSONObject.NULL)
        put("max_frame_duration_ns", get(CameraCharacteristics.SENSOR_INFO_MAX_FRAME_DURATION) ?: JSONObject.NULL)
        put("flash_available", get(CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false)
        put("ae_lock_available", get(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE) ?: false)
        put("awb_lock_available", get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE) ?: false)
        put("ae_modes", jsonInts(get(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES), Names::aeMode))
        put("af_modes", jsonInts(get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES), Names::afMode))
        put("noise_reduction_modes", jsonInts(get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES), Names::noiseReduction))
        put("edge_modes", jsonInts(get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES), Names::edge))
        put("video_stabilization_modes", jsonInts(get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES), Names::videoStabilization))
        put("optical_stabilization_modes", jsonInts(get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION), Names::opticalStabilization))
        put("focal_lengths_mm", JSONArray().apply { get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.forEach { put(num(it.toDouble(), 2)) } })
        put("sensor_orientation", get(CameraCharacteristics.SENSOR_ORIENTATION) ?: JSONObject.NULL)
        put("physical_camera_ids", JSONArray(physical))
    }
}

fun describeDevice(): JSONObject = JSONObject().apply {
    put("manufacturer", Build.MANUFACTURER)
    put("brand", Build.BRAND)
    put("model", Build.MODEL)
    put("device", Build.DEVICE)
    put("product", Build.PRODUCT)
    put("hardware", Build.HARDWARE)
    put("soc_model", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else JSONObject.NULL)
    put("android_release", Build.VERSION.RELEASE)
    put("sdk_int", Build.VERSION.SDK_INT)
    put("build_display", Build.DISPLAY)
}

fun describeApp(context: Context): JSONObject {
    val pm = context.packageManager
    val info = if (Build.VERSION.SDK_INT >= 33) {
        pm.getPackageInfo(context.packageName, android.content.pm.PackageManager.PackageInfoFlags.of(0))
    } else {
        @Suppress("DEPRECATION")
        pm.getPackageInfo(context.packageName, 0)
    }
    return JSONObject().apply {
        put("package", context.packageName)
        put("version", info.versionName ?: "")
        put("version_code", info.longVersionCode)
    }
}
