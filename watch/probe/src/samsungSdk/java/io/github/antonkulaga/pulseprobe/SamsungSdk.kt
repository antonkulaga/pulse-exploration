package io.github.antonkulaga.pulseprobe

import android.content.Context
import android.os.Handler
import android.util.Log
import com.samsung.android.service.health.tracking.ConnectionListener
import com.samsung.android.service.health.tracking.HealthTracker
import com.samsung.android.service.health.tracking.HealthTrackerException
import com.samsung.android.service.health.tracking.HealthTrackingService
import com.samsung.android.service.health.tracking.data.DataPoint
import com.samsung.android.service.health.tracking.data.HealthTrackerType
import com.samsung.android.service.health.tracking.data.PpgType
import com.samsung.android.service.health.tracking.data.ValueKey
import java.io.BufferedWriter
import java.io.File

/** Compiled when watch/probe/libs holds the Samsung Health Sensor SDK .aar (see build.gradle.kts). */
fun openSamsungHealth(context: Context): SamsungHealth = SamsungSdk(context)

/**
 * The real Health Platform connection. The SDK keeps one process-wide connector whose listener is the
 * last one set, so only one of these should be connected at a time (the service runs one thing at a time).
 * Connection callbacks arrive on the main thread, tracker callbacks on binder threads.
 */
class SamsungSdk(private val context: Context) : SamsungHealth, ConnectionListener {
    @Volatile override var installed = false
        private set
    @Volatile override var connected = false
        private set
    @Volatile override var error = ""
        private set
    @Volatile override var reason = ""
        private set
    @Volatile override var version = ""
        private set
    @Volatile override var supported: List<String> = emptyList()
        private set
    private var service: HealthTrackingService? = null
    private var onResult: (() -> Unit)? = null
    private var answered = false

    override fun connect(onResult: () -> Unit) {
        this.onResult = onResult
        val platform = healthPlatformVersion(context)
        installed = platform != null
        if (platform == null) return fail("$HEALTH_PLATFORM_PACKAGE not installed", REASON_NOT_INSTALLED)
        try {
            // connectService reports a missing or old platform synchronously through onConnectionFailed.
            val s = HealthTrackingService(this, context)
            service = s
            s.connectService()
        } catch (e: Throwable) {
            if (e is VirtualMachineError) throw e
            fail("connectService: $e", REASON_NOT_CONNECTED)
        }
    }

    override fun onConnectionSuccess() {
        synchronized(this) {
            if (answered) return
            try {
                val capability = service!!.trackingCapability
                supported = capability.supportHealthTrackerTypes.map { it.name }
                version = capability.version.orEmpty()
            } catch (e: Exception) {
                error = clean("getTrackingCapability: $e")
            }
            connected = true
        }
        answer()
    }

    override fun onConnectionFailed(e: HealthTrackerException) {
        val code = try {
            e.errorCode
        } catch (_: Exception) {
            -1
        }
        when (code) {
            HealthTrackerException.PACKAGE_NOT_INSTALLED ->
                fail("connection failed: PACKAGE_NOT_INSTALLED", REASON_NOT_INSTALLED, notInstalled = true)
            HealthTrackerException.OLD_PLATFORM_VERSION -> fail(
                "connection failed: OLD_PLATFORM_VERSION (Health Platform ${healthPlatformVersion(context)})",
                REASON_OLD_PLATFORM, notInstalled = true,
            )
            else -> fail("connection failed: code=$code ${e.message}", REASON_NOT_CONNECTED)
        }
    }

    override fun onConnectionEnded() {}

    override fun timeout() = fail("no answer from Health Platform", REASON_NOT_CONNECTED)

    private fun fail(what: String, why: String, notInstalled: Boolean = false) {
        synchronized(this) {
            if (answered) return
            error = clean(what)
            reason = why
            if (notInstalled) installed = false
        }
        answer()
    }

    private fun answer() {
        val callback = synchronized(this) {
            if (answered) return
            answered = true
            onResult
        }
        callback?.invoke()
    }

    override fun checkPolicy(type: String, waitMs: Long, handler: Handler, onResult: (String, String) -> Unit) {
        val s = service
        if (s == null || !connected) {
            handler.post { onResult(POLICY_UNKNOWN, "not connected") }
            return
        }
        var tracker: HealthTracker? = null
        var finished = false
        var permissionError = false
        lateinit var deadline: Runnable
        // Runs on handler only.
        fun finish(policy: String, why: String) {
            if (finished) return
            finished = true
            handler.removeCallbacks(deadline)
            try {
                tracker?.unsetEventListener()
            } catch (_: Exception) {
            }
            onResult(policy, why)
        }
        deadline = Runnable {
            if (permissionError) finish(POLICY_UNKNOWN, "PERMISSION_ERROR on $type") else finish(POLICY_ALLOWED, "")
        }
        val listener = object : HealthTracker.TrackerEventListener {
            override fun onDataReceived(points: List<DataPoint>) {
                handler.post { finish(POLICY_ALLOWED, "") }
            }

            override fun onFlushCompleted() {}

            override fun onError(trackerError: HealthTracker.TrackerError) {
                handler.post {
                    when (trackerError) {
                        HealthTracker.TrackerError.SDK_POLICY_ERROR -> finish(POLICY_DENIED, "SDK_POLICY_ERROR on $type")
                        // A permission problem, not the policy: keep waiting in case the policy error follows.
                        else -> permissionError = true
                    }
                }
            }
        }
        try {
            val t = s.getHealthTracker(HealthTrackerType.valueOf(type))
            tracker = t
            t.setEventListener(listener)
        } catch (e: Throwable) {
            if (e is VirtualMachineError) throw e
            handler.post { finish(POLICY_UNKNOWN, clean("policy check on $type: $e")) }
            return
        }
        handler.postDelayed(deadline, waitMs)
    }

    override fun source(tracker: SdkTracker, file: File?): Source =
        COLUMNS[tracker.alias]?.let { SamsungSource(tracker, it, this, file) }
            ?: UnavailableSource(tracker.alias, Tier.FULL, "no CSV columns for ${tracker.alias}")

    internal fun tracker(type: String, ppg: Set<PpgType>?): HealthTracker {
        val s = service ?: throw IllegalStateException("not connected")
        val t = HealthTrackerType.valueOf(type)
        return if (ppg != null) s.getHealthTracker(t, ppg) else s.getHealthTracker(t)
    }

    override fun disconnect() {
        try {
            service?.disconnectService()
        } catch (_: Exception) {
        }
    }

    /** CSV columns and PPG channels per tracker alias. */
    class Columns(val columns: List<Pair<String, ValueKey<*>>>, val ppg: Set<PpgType>? = null)

    companion object {
        private val ppgColumns = listOf(
            "green" to ValueKey.PpgSet.PPG_GREEN, "green_status" to ValueKey.PpgSet.GREEN_STATUS,
            "ir" to ValueKey.PpgSet.PPG_IR, "ir_status" to ValueKey.PpgSet.IR_STATUS,
            "red" to ValueKey.PpgSet.PPG_RED, "red_status" to ValueKey.PpgSet.RED_STATUS,
        )
        private val allPpg = setOf(PpgType.GREEN, PpgType.IR, PpgType.RED)

        val COLUMNS: Map<String, Columns> = mapOf(
            "sdk_ppg" to Columns(ppgColumns, allPpg),
            "sdk_ppg100" to Columns(ppgColumns, allPpg),
            "sdk_hr" to Columns(
                listOf(
                    "hr" to ValueKey.HeartRateSet.HEART_RATE, "hr_status" to ValueKey.HeartRateSet.HEART_RATE_STATUS,
                    "ibi_ms" to ValueKey.HeartRateSet.IBI_LIST, "ibi_status" to ValueKey.HeartRateSet.IBI_STATUS_LIST,
                )
            ),
            "sdk_acc" to Columns(
                listOf(
                    "x" to ValueKey.AccelerometerSet.ACCELEROMETER_X, "y" to ValueKey.AccelerometerSet.ACCELEROMETER_Y,
                    "z" to ValueKey.AccelerometerSet.ACCELEROMETER_Z,
                )
            ),
            "sdk_skin" to Columns(
                listOf(
                    "object_c" to ValueKey.SkinTemperatureSet.OBJECT_TEMPERATURE,
                    "ambient_c" to ValueKey.SkinTemperatureSet.AMBIENT_TEMPERATURE,
                    "status" to ValueKey.SkinTemperatureSet.STATUS,
                )
            ),
            "sdk_eda" to Columns(
                listOf("skin_conductance" to ValueKey.EdaSet.SKIN_CONDUCTANCE, "status" to ValueKey.EdaSet.STATUS)
            ),
            "sdk_ecg" to Columns(
                listOf(
                    "ecg_mv" to ValueKey.EcgSet.ECG_MV, "lead_off" to ValueKey.EcgSet.LEAD_OFF,
                    "sequence" to ValueKey.EcgSet.SEQUENCE, "ppg_green" to ValueKey.EcgSet.PPG_GREEN,
                    "max_threshold_mv" to ValueKey.EcgSet.MAX_THRESHOLD_MV,
                    "min_threshold_mv" to ValueKey.EcgSet.MIN_THRESHOLD_MV,
                )
            ),
            "sdk_spo2" to Columns(
                listOf(
                    "spo2" to ValueKey.SpO2Set.SPO2, "hr" to ValueKey.SpO2Set.HEART_RATE,
                    "status" to ValueKey.SpO2Set.STATUS, "accuracy_flag" to ValueKey.SpO2Set.ACCURACY_FLAG,
                )
            ),
        )
    }
}

/** Tier "all": a Samsung Health Sensor SDK tracker. Data arrives in batches; wall_ms is batch arrival. */
class SamsungSource(
    private val spec: SdkTracker,
    private val columns: SamsungSdk.Columns,
    private val sdk: SamsungSdk,
    private val file: File?,
) : Source, HealthTracker.TrackerEventListener {
    override val alias = spec.alias
    override val tier = Tier.FULL
    private var out: BufferedWriter? = null
    private val stats = Stats(1e3)
    private var tracker: HealthTracker? = null
    private var registered = false
    @Volatile override var error = ""
        private set
    @Volatile override var sdkFailed = false
        private set
    override val events get() = stats.events
    private var closed = false
    /** ECG only: whether a finger is on the key, from LEAD_OFF (0 on, 5 off, -1 inside a 10-sample bundle). */
    private var leadOn: Boolean? = null

    override fun start() {
        when {
            !sdk.connected -> fail(sdk.error.ifEmpty { "SDK not connected" })
            spec.type !in sdk.supported -> error = REASON_UNSUPPORTED
            else -> try {
                val t = sdk.tracker(spec.type, columns.ppg)
                t.setEventListener(this)
                tracker = t
                registered = true
            } catch (e: Throwable) {
                if (e is VirtualMachineError) throw e
                fail(clean(e.toString()))
            }
        }
    }

    private fun fail(why: String) {
        if (error.isEmpty()) error = why
        sdkFailed = true
    }

    @Synchronized
    override fun onDataReceived(points: List<DataPoint>) {
        if (closed) return
        val wall = System.currentTimeMillis()
        try {
            for (point in points) {
                val values = columns.columns.joinToString(",") { (_, key) -> format(point, key) }
                if (stats.events == 0) {
                    stats.valuesPerEvent = columns.columns.size
                    stats.firstValues = values
                    out = file?.bufferedWriter()
                    out?.write("timestamp_ms,wall_ms," + columns.columns.joinToString(",") { it.first } + "\n")
                }
                out?.write("${point.timestamp},$wall,$values\n")
                stats.add(point.timestamp)
                if (spec.type == "ECG_ON_DEMAND") showLead(point)
            }
        } catch (e: Exception) {
            if (error.isEmpty()) error = clean("write: $e")
        }
    }

    /** ECG data arrives every ~20 ms, so the watch can tell the user at once whether the key is touched. */
    private fun showLead(point: DataPoint) {
        val flag = point.getValue(ValueKey.EcgSet.LEAD_OFF) ?: return
        if (flag < 0) return
        val on = flag == 0
        if (on == leadOn) return
        leadOn = on
        Log.i(RecorderService.TAG, "ecg lead ${if (on) "on" else "off"} (lead_off=$flag)")
        ProbeState.activity(if (on) "ECG: contact, keep still" else "ECG: touch the top key")
    }

    override fun onFlushCompleted() {}

    /** PERMISSION_ERROR or SDK_POLICY_ERROR; the latter means developer mode is off for this app. */
    override fun onError(trackerError: HealthTracker.TrackerError) = fail(trackerError.name)

    @Suppress("UNCHECKED_CAST")
    private fun format(point: DataPoint, key: ValueKey<*>): String = when (val v = point.getValue(key as ValueKey<Any?>)) {
        null -> ""
        is List<*> -> v.joinToString("|")
        else -> v.toString()
    }

    /** Asks for the pending batch; the listener stays attached until [close] so the flushed data arrives. */
    override fun stop() {
        try {
            tracker?.flush()
        } catch (_: Exception) {
        }
    }

    @Synchronized
    override fun close() {
        try {
            tracker?.unsetEventListener()
        } catch (_: Exception) {
        }
        closed = true
        try {
            out?.close()
        } catch (_: Exception) {
        }
    }

    override fun row() = listOf(
        tier.label, alias, spec.type in sdk.supported, registered, stats.events, stats.rateHz(), stats.valuesPerEvent,
        stats.firstValues, error,
    ).joinToString("\t")
}
