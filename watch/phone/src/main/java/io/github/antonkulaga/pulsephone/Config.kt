package io.github.antonkulaga.pulsephone

import android.content.Context
import android.graphics.RectF

/** Where the signal comes from. [wire] is the server's `mode` value. */
enum class Mode(val wire: String, val label: String) {
    FINGER("finger", "Fingertip"),
    FACE("face", "Face");

    companion object {
        fun parse(text: String?): Mode? = when (text?.trim()?.lowercase()) {
            "finger", "fingertip" -> FINGER
            "face" -> FACE
            else -> null
        }
    }
}

/** A fixed sensor exposure. Only applied when the camera lists the MANUAL_SENSOR capability. */
data class ManualExposure(val exposureNs: Long, val iso: Int)

/** Everything a recording depends on; a camera bound with an equal config is reused as is. */
data class RecordingConfig(val mode: Mode, val seconds: Int, val try60: Boolean, val manual: ManualExposure?)

/** Regions the colour means are taken over, as fractions of the upright (portrait) image. */
object Roi {
    /**
     * Finger: the central half of the full frame in each direction. Under a fingertip the whole frame
     * is red; the centre skips the vignetted corners and the rim where the finger lifts off the lens.
     */
    val FINGER = RectF(0.25f, 0.25f, 0.75f, 0.75f)

    /**
     * Face: forehead and upper cheeks, as fractions of the visible preview (the analysis crop rect,
     * which a shared ViewPort makes match the preview). Horizontally centred, so the mirrored front
     * preview and the unmirrored analysis frame agree without flipping.
     */
    val FACE = RectF(0.28f, 0.20f, 0.72f, 0.52f)
}

/** Persisted settings. Intent extras write through to them, so an adb run also sets the defaults. */
class Prefs(context: Context) {
    private val sp = context.getSharedPreferences("pulsephone", Context.MODE_PRIVATE)

    var server: String
        get() = sp.getString("server", null)?.takeIf { it.isNotBlank() } ?: DEFAULT_SERVER
        set(value) = sp.edit().putString("server", value.trim()).apply()

    var mode: Mode
        get() = Mode.parse(sp.getString("mode", null)) ?: Mode.FINGER
        set(value) = sp.edit().putString("mode", value.wire).apply()

    var seconds: Int
        get() = sp.getInt("seconds", DEFAULT_SECONDS).coerceIn(MIN_SECONDS, MAX_SECONDS)
        set(value) = sp.edit().putInt("seconds", value.coerceIn(MIN_SECONDS, MAX_SECONDS)).apply()

    var try60: Boolean
        get() = sp.getBoolean("try60", false)
        set(value) = sp.edit().putBoolean("try60", value).apply()

    var manual: Boolean
        get() = sp.getBoolean("manual", false)
        set(value) = sp.edit().putBoolean("manual", value).apply()

    var exposureUs: Int
        get() = sp.getInt("exposure_us", 8_000)
        set(value) = sp.edit().putInt("exposure_us", value).apply()

    var iso: Int
        get() = sp.getInt("iso", 100)
        set(value) = sp.edit().putInt("iso", value).apply()

    companion object {
        const val DEFAULT_SERVER = "http://192.168.1.177:8765"
        const val MIN_SECONDS = 20
        const val MAX_SECONDS = 120
        const val DEFAULT_SECONDS = 45
    }
}
