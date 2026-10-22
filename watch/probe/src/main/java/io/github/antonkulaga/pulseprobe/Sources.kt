package io.github.antonkulaga.pulseprobe

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import java.io.BufferedWriter
import java.io.File
import java.util.Locale

/**
 * What a user has to do before a signal works.
 *
 * - [BASIC] ("25hz"): plain Android sensors. Install the app and grant the heart-rate permission.
 * - [FULL] ("all"): the Samsung Health Sensor SDK. Each user also has to turn on Health Sensor Service
 *   developer mode on their watch (or we need Samsung partner approval).
 * - [BLOCKED]: visible but reserved for Samsung's own apps (signature permission).
 */
enum class Tier(val label: String) { BASIC("25hz"), FULL("all"), BLOCKED("blocked") }

/**
 * One signal being probed or recorded. Implementations write one CSV each when given a file; the file is
 * created on the first event, so a source that delivers nothing leaves no (empty, unparseable) CSV.
 * None of the methods throw: a failure ends up in [error].
 */
interface Source {
    val alias: String
    val tier: Tier
    /** Events delivered so far. */
    val events: Int
    /** Why the source could not (fully) record; empty when it worked. */
    val error: String
    /** True when the Samsung Health Platform itself failed (connection, policy, permission): worth a re-probe. */
    val sdkFailed: Boolean get() = false
    fun start()
    fun stop()
    fun close()

    /** tier, alias, available, registered, events, rate_hz, values_per_event, first_values, error */
    fun row(): String
}

/** A source that cannot be opened at all, such as a tier "all" one in an app built without the SDK. */
class UnavailableSource(override val alias: String, override val tier: Tier, override val error: String) : Source {
    override val events = 0
    override fun start() {}
    override fun stop() {}
    override fun close() {}
    override fun row() = listOf(tier.label, alias, false, false, 0, "", 0, "", error).joinToString("\t")
}

/** Looks a sensor up by its string type (Samsung types are not public constants), then by numeric type. */
fun findSensor(sensorManager: SensorManager, stringType: String, type: Int): Sensor? =
    sensorManager.getSensorList(Sensor.TYPE_ALL)
        .filter { it.stringType == stringType }
        .minByOrNull { if (it.isWakeUpSensor) 1 else 0 }
        ?: sensorManager.getDefaultSensor(type)

/** Counts events and measures the effective rate from the source's own timestamps. */
class Stats(private val timestampsPerSecond: Double) {
    var events = 0
        private set
    var valuesPerEvent = 0
    var firstValues = ""
    private var first = 0L
    private var last = 0L

    fun add(timestamp: Long) {
        if (events == 0) first = timestamp
        last = timestamp
        events++
    }

    fun rateHz(): String =
        if (events < 2 || last <= first) "" else "%.2f".format(Locale.US, (events - 1) * timestampsPerSecond / (last - first))
}

internal fun clean(text: String?) = text.orEmpty().replace('\t', ' ').replace('\n', ' ')

/** Tier "25hz": an Android SensorManager sensor at its fastest rate. */
class AndroidSensorSource(
    override val alias: String,
    override val tier: Tier,
    private val sensorManager: SensorManager,
    private val sensor: Sensor?,
    private val handler: Handler,
    private val file: File?,
    /** Samsung's raw HR sensor packs integer ADC counts into the float fields (1.4E-45 = int 1). */
    private val packedInts: Boolean,
) : Source, SensorEventListener {
    private var out: BufferedWriter? = null
    private val stats = Stats(1e9)
    private var registered = false
    @Volatile override var error = ""
        private set
    override val events get() = stats.events
    private var closed = false

    override fun start() {
        val s = sensor ?: return
        registered = try {
            sensorManager.registerListener(this, s, SensorManager.SENSOR_DELAY_FASTEST, 0, handler)
        } catch (e: Exception) {
            error = clean(e.message ?: e.toString())
            false
        }
        if (!registered && error.isEmpty()) error = "registerListener returned false"
    }

    @Synchronized
    override fun onSensorChanged(event: SensorEvent) {
        if (closed) return
        val values = format(event.values)
        try {
            if (stats.events == 0) {
                stats.valuesPerEvent = event.values.size
                stats.firstValues = values
                out = file?.bufferedWriter()
                out?.write("event_ns,wall_ms,accuracy," + event.values.indices.joinToString(",") { "v$it" } + "\n")
            }
            out?.write("${event.timestamp},${System.currentTimeMillis()},${event.accuracy},$values\n")
        } catch (e: java.io.IOException) {
            if (error.isEmpty()) error = clean("write: $e")
        }
        stats.add(event.timestamp)
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}

    private fun format(values: FloatArray) =
        if (packedInts) values.joinToString(",") { java.lang.Float.floatToRawIntBits(it).toString() }
        else values.joinToString(",")

    override fun stop() {
        if (registered) sensorManager.unregisterListener(this)
    }

    @Synchronized
    override fun close() {
        closed = true
        try {
            out?.close()
        } catch (_: java.io.IOException) {
        }
    }

    override fun row() = listOf(
        tier.label, alias, sensor != null, registered, stats.events, stats.rateHz(), stats.valuesPerEvent,
        stats.firstValues, error,
    ).joinToString("\t")
}

object Catalog {
    /** alias → (string type, numeric type, tier), as listed by `dumpsys sensorservice` on SM-R965F. */
    val android = linkedMapOf(
        "hr_raw" to Triple("com.samsung.sensor.hr_raw", 69682, Tier.BASIC),
        "hr" to Triple(Sensor.STRING_TYPE_HEART_RATE, Sensor.TYPE_HEART_RATE, Tier.BASIC),
        "acc" to Triple(Sensor.STRING_TYPE_ACCELEROMETER, Sensor.TYPE_ACCELEROMETER, Tier.BASIC),
        "gyro" to Triple(Sensor.STRING_TYPE_GYROSCOPE, Sensor.TYPE_GYROSCOPE, Tier.BASIC),
        "hr_raw_fac" to Triple("com.samsung.sensor.hr_raw_fac", 69683, Tier.BLOCKED),
        "hr_raw_fac2" to Triple("com.samsung.sensor.hr_raw_fac2", 69684, Tier.BLOCKED),
        "ecg_raw" to Triple("com.samsung.sensor.ecg", 69669, Tier.BLOCKED),
        "skin_temp_raw" to Triple("com.samsung.sensor.skin_temp", 69686, Tier.BLOCKED),
        "acc32g" to Triple("com.samsung.sensor.accelerometer_32g", 69657, Tier.BLOCKED),
    )

    /** Vendor sensors whose float fields carry integer bit patterns. */
    val packedInts = setOf("hr_raw", "hr_raw_fac", "hr_raw_fac2")

    /** Tier "all" trackers by alias; [SamsungHealth] knows their CSV columns. */
    val samsung: Map<String, SdkTracker> = SDK_TRACKERS

    /** Everything a probe opens: both tiers' continuous signals. On-demand ones need the user's attention. */
    val probe = android.keys.toList() + samsung.values.filterNot { it.onDemand }.map { it.alias }
}
