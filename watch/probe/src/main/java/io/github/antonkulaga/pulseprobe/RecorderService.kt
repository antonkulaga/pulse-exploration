package io.github.antonkulaga.pulseprobe

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Foreground service of type "health" that detects what the watch can record and records it. An activity
 * alone is not enough: once the watch goes to ambient, Wear OS treats the app as idle, disables its wake
 * lock and stops delivering sensor events.
 *
 * Commands (extra "cmd"):
 * - session: records features (extra "features", default ppg,hr,ibi,motion) for "seconds" (default 60),
 *   each through every source this watch allows: Samsung SDK ones in tier "all", Android ones always.
 * - spot: one on-demand feature (extra "feature": ppg100, ecg or spo2), alone, at most 30 s. Refused,
 *   with a run directory saying why, when the watch does not allow it.
 * - caps: re-checks the watch and saves the result as a run.
 * - probe, record: the original exploration commands (all signals / explicit "sensors" list).
 * - wifi: holds Wi-Fi for "seconds".
 * - detect: re-checks the watch without saving a run (the activity uses it).
 *
 * Every run writes filesDir/runs/<stamp>-<kind>/: capabilities.json, one CSV per source that delivered
 * data, <kind>.tsv with one row per source, sdk.tsv when the SDK was used, and meta.txt. When a server
 * is set, the run is then uploaded to it over Wi-Fi.
 */
class RecorderService : Service() {

    private lateinit var sensorManager: SensorManager
    private val thread = HandlerThread("sensors").apply { start() }
    private val handler = Handler(thread.looper)
    private val uploadThread = HandlerThread("upload").apply { start() }
    private val uploadHandler = Handler(uploadThread.looper)
    private val wakeLock by lazy {
        getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "PulseProbe:record")
            .apply { setReferenceCounted(false) }
    }
    private val wifi by lazy { WifiHold(this) }
    private val wifiRelease = Runnable { wifi.release(); log("wifi hold ended"); stopIfIdle() }
    private val detector by lazy { CapabilityProbe(this, sensorManager, handler) }
    /** What this watch can record, cached for the life of this service; re-probed by caps or an SDK failure. */
    private var capabilities: Capabilities? = null
    @Volatile private var busy = false

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(SensorManager::class.java)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val cmd = intent?.getStringExtra(EXTRA_CMD) ?: return START_NOT_STICKY
        intent.getStringExtra(EXTRA_SERVER)?.let { server = it.trim() }
        val seconds = intent.getIntExtra(EXTRA_SECONDS, defaultSeconds(cmd))
        goForeground(if (cmd == "caps" || cmd == "detect") cmd else "$cmd ${seconds}s")
        if (cmd == "wifi") {
            holdWifi(seconds)
            return START_NOT_STICKY
        }
        if (cmd !in RUN_COMMANDS) {
            log("unknown command $cmd")
            stopIfIdle()
            return START_NOT_STICKY
        }
        if (busy) {
            log("busy, ignoring $cmd")
            return START_NOT_STICKY
        }
        busy = true
        ProbeState.activity(if (cmd == "caps" || cmd == "detect") "checking the watch" else "$cmd starting")
        val sensors = intent.getStringExtra(EXTRA_SENSORS).list()
        val features = intent.getStringExtra(EXTRA_FEATURES).list()
        val feature = intent.getStringExtra(EXTRA_FEATURE)?.trim().orEmpty()
        handler.post {
            guarded(cmd) {
                wakeLock.acquire(DETECT_WAKE_MS)
                when (cmd) {
                    "probe" -> record("probe", Catalog.probe, seconds, writeCsv = false, plan = null)
                    "record" -> record("record", sensors ?: listOf("hr_raw", "hr", "acc"), seconds, writeCsv = true, plan = null)
                    "session" -> session(features ?: DEFAULT_SESSION_FEATURES, seconds.coerceAtLeast(1))
                    "spot" -> spot(feature, seconds)
                    "caps" -> caps()
                    "detect" -> detect { caps -> logFeatures(caps); done() }
                }
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        wifi.release()
        thread.quitSafely()
        uploadThread.quitSafely()
        if (wakeLock.isHeld) wakeLock.release()
        super.onDestroy()
    }

    /** Keeps Wi-Fi (and so wireless adb) up for [seconds]; 0 releases it. */
    private fun holdWifi(seconds: Int) {
        handler.removeCallbacks(wifiRelease)
        if (seconds <= 0) return wifiRelease.run()
        wifi.acquire()
        handler.postAtTime(wifiRelease, SystemClock.uptimeMillis() + seconds * 1000L)
        log("holding wifi for $seconds s")
    }

    /** Re-probes the watch, caches the answer for this service and shows it in the activity. */
    private fun detect(then: (Capabilities) -> Unit) = detector.detect { caps ->
        capabilities = caps
        guarded("after detection") { then(caps) }
    }

    /** Uses the cached capabilities, or detects them first ([block]'s second argument is then true). */
    private fun withCapabilities(block: (Capabilities, Boolean) -> Unit) {
        val cached = capabilities
        if (cached != null) return block(cached, false)
        ProbeState.activity("checking the watch")
        detect { block(it, true) }
    }

    /** Records every source of every available [features] at once. */
    private fun session(features: List<String>, seconds: Int) = withCapabilities { caps, fresh ->
        val aliases = mutableListOf<String>()
        for (name in features) {
            val spec = FEATURE_SPECS[name]
            val f = caps.features[name]
            when {
                spec == null || f == null -> log("unknown feature '$name' ignored")
                spec.onDemand -> log("feature $name is on-demand and runs alone: use cmd spot --es feature $name")
                !f.available -> log("feature $name unavailable: ${f.reason}")
                else -> aliases += f.sources
            }
        }
        val sources = aliases.distinct()
        if (sources.isEmpty()) return@withCapabilities refuse("session", caps, emptyList(), "session: nothing to record for $features")
        log("session tier ${caps.tier}: ${features.joinToString(",")} → ${sources.joinToString(",")}")
        record("session", sources, seconds, writeCsv = true, plan = caps, settleMs = settle(fresh, sources))
    }

    /** One on-demand feature, alone. */
    private fun spot(feature: String, seconds: Int) {
        val spec = FEATURE_SPECS[feature]
        if (spec == null || !spec.onDemand) {
            log("unknown spot feature '$feature' ignored (use ppg100, ecg or spo2)")
            return done()
        }
        val secs = seconds.coerceIn(1, SPOT_MAX_SECONDS)
        if (secs != seconds) log("spot runs at most $SPOT_MAX_SECONDS s; recording $secs s")
        withCapabilities { caps, fresh ->
            val f = caps.features.getValue(feature)
            if (!f.available) return@withCapabilities refuse("spot", caps, spec.candidates, "spot $feature refused: ${f.reason}")
            log("spot $feature (tier ${caps.tier}) → ${f.sources.joinToString(",")}")
            record("spot", f.sources, secs, writeCsv = true, plan = caps, settleMs = settle(fresh, f.sources))
        }
    }

    /** After a fresh detection, give the Health Platform a moment before opening trackers again. */
    private fun settle(fresh: Boolean, sources: List<String>) = if (fresh && sources.any { it in SDK_TRACKERS }) SETTLE_MS else 0L

    private fun caps() {
        if (server != null) wifi.acquire()
        val dir = newRunDir("caps")
        val start = System.currentTimeMillis()
        detect { caps ->
            logFeatures(caps)
            writeCapabilities(dir, caps)
            writeSummary(dir, "caps", caps.featureSources().map(caps::unopenedRow))
            writeMeta(dir, start, System.currentTimeMillis(), 0)
            log("caps done → ${dir.name}")
            upload(dir)
        }
    }

    /** A run that records nothing, so the laptop still learns why. */
    private fun refuse(kind: String, caps: Capabilities, aliases: List<String>, why: String) {
        log(why)
        if (server != null) wifi.acquire()
        val dir = newRunDir(kind)
        val now = System.currentTimeMillis()
        writeCapabilities(dir, caps)
        writeSummary(dir, kind, aliases.map(caps::unopenedRow))
        writeMeta(dir, now, now, 0)
        log("$kind refused → ${dir.name}")
        upload(dir)
    }

    private class Run(
        val kind: String,
        val dir: File,
        val sources: List<Source>,
        val seconds: Int,
        val sdk: SamsungHealth?,
        /** The capabilities a session or spot was resolved with; null for probe and record. */
        val plan: Capabilities?,
    ) {
        var startWall = 0L
    }

    /**
     * Opens every requested signal for [seconds]: tier "25hz" through SensorManager, tier "all" through
     * one Health Platform connection. A probe only counts events; the other kinds write them to CSV.
     * The Android sources record whatever happens to the SDK connection.
     */
    private fun record(kind: String, aliases: List<String>, seconds: Int, writeCsv: Boolean, plan: Capabilities?, settleMs: Long = 0) {
        wakeLock.acquire((seconds + 90) * 1000L)
        if (server != null) wifi.acquire()
        val dir = newRunDir(kind)
        if (kind == "probe") {
            writePermissions(File(dir, "permissions.tsv"))
            writeSensorList(File(dir, "sensors.tsv"))
        }
        val unknown = aliases.filter { it !in Catalog.android && it !in Catalog.samsung }
        if (unknown.isNotEmpty()) log("unknown signals ignored: $unknown")

        val sdk = if (aliases.any { it in Catalog.samsung }) {
            try {
                openSamsungHealth(this)
            } catch (e: Throwable) {
                if (e is VirtualMachineError) throw e
                Log.w(TAG, "openSamsungHealth failed", e)
                null
            }
        } else null
        val sources = aliases.mapNotNull { alias ->
            val file = if (writeCsv) File(dir, "$alias.csv") else null
            Catalog.android[alias]?.let { (stringType, type, tier) ->
                AndroidSensorSource(alias, tier, sensorManager, findSensor(sensorManager, stringType, type), handler, file, alias in Catalog.packedInts)
            } ?: Catalog.samsung[alias]?.let { tracker ->
                sdk?.source(tracker, file) ?: UnavailableSource(alias, Tier.FULL, "cannot load the Samsung SDK")
            }
        }
        log("$kind ${sources.joinToString { "${it.alias}[${it.tier.label}]" }} for $seconds s → ${dir.name}")
        ProbeState.activity("$kind ${seconds}s → ${dir.name}")
        val run = Run(kind, dir, sources, seconds, sdk, plan)

        handler.postDelayed({
            guarded("$kind start") {
                if (sdk == null) return@guarded startAll(run)
                var started = false
                val timeout = Runnable { sdk.timeout() }
                sdk.connect {
                    handler.post {
                        guarded("$kind start") {
                            if (started) return@guarded
                            started = true
                            handler.removeCallbacks(timeout)
                            if (!sdk.connected) log("sdk not connected: ${sdk.error}; recording the Android sources only")
                            startAll(run)
                        }
                    }
                }
                handler.postDelayed(timeout, SDK_TIMEOUT_MS)
            }
        }, settleMs)
    }

    private fun startAll(run: Run) {
        run.sources.forEach { guard(it, "start") { it.start() } }
        run.startWall = System.currentTimeMillis()
        // postAtTime on the uptime clock is safe here because the wake lock keeps the CPU awake.
        handler.postAtTime({ guarded("${run.kind} stop") { stopAll(run) } }, SystemClock.uptimeMillis() + run.seconds * 1000L)
    }

    private fun stopAll(run: Run) {
        run.sources.forEach { guard(it, "stop") { it.stop() } }
        // Samsung delivers SDK data in batches; give the flushed batch time to arrive before closing.
        handler.postDelayed({ guarded("${run.kind} finish") { finish(run) } }, FLUSH_WAIT_MS)
    }

    private fun finish(run: Run) {
        run.sources.forEach { guard(it, "close") { it.close() } }
        val rows = run.sources.map(Source::row)
        writeSummary(run.dir, run.kind, rows)
        run.sdk?.let { s ->
            File(run.dir, "sdk.tsv").writeText(
                "connected\terror\tversion\tsupported\n" +
                    "${s.connected}\t${s.error}\t${s.version}\t${s.supported.joinToString(",")}\n"
            )
            log("sdk connected=${s.connected} version=${s.version} error=${s.error} supported=${s.supported}")
            s.disconnect()
        }
        writeMeta(run.dir, run.startWall, System.currentTimeMillis(), run.seconds)
        rows.forEach { log(it) }
        log("${run.kind} done → ${run.dir.name}")

        val sdkFailed = run.sources.any { it.sdkFailed }
        val cached = capabilities
        val plan = run.plan
        when {
            plan != null -> {
                writeCapabilities(run.dir, plan.afterRun(run.sources.associate { it.alias to (it.events to it.error) }))
                if (!sdkFailed) return upload(run.dir)
                log("the Samsung SDK failed during ${run.kind}; checking the watch again")
                detect { caps -> logFeatures(caps); upload(run.dir) }
            }
            sdkFailed || cached == null -> detect { caps -> writeCapabilities(run.dir, caps); upload(run.dir) }
            else -> {
                writeCapabilities(run.dir, cached)
                upload(run.dir)
            }
        }
    }

    /** An unexpected failure in a later stage ends the command instead of killing the process. */
    private inline fun guarded(what: String, block: () -> Unit) {
        try {
            block()
        } catch (e: Exception) {
            Log.e(TAG, "$what failed", e)
            done()
        }
    }

    /** A source's own failure must never stop the others. */
    private inline fun guard(source: Source, what: String, action: () -> Unit) {
        try {
            action()
        } catch (e: Exception) {
            Log.w(TAG, "${source.alias} $what failed", e)
        }
    }

    private fun upload(dir: File) {
        val target = server ?: return done()
        ProbeState.activity("uploading ${dir.name}")
        uploadHandler.post {
            log(Uploader.upload(target, dir, wifi.network))
            handler.post { done() }
        }
    }

    private fun done() {
        busy = false
        if (ProbeState.detecting) ProbeState.detecting(false)
        ProbeState.activity("")
        if (!handler.hasCallbacks(wifiRelease)) wifi.release()
        if (wakeLock.isHeld) wakeLock.release()
        stopIfIdle()
    }

    private fun stopIfIdle() {
        if (busy || wifi.held) return
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun logFeatures(caps: Capabilities) {
        log("tier ${caps.tier}: sdk installed=${caps.sdk.installed} connected=${caps.sdk.connected} policy=${caps.sdk.policy}")
        FEATURES.forEach { log("feature " + caps.describe(it.name)) }
    }

    private fun writeCapabilities(dir: File, caps: Capabilities) = File(dir, "capabilities.json").writeText(caps.toJson())

    private fun writeSummary(dir: File, kind: String, rows: List<String>) =
        File(dir, "$kind.tsv").writeText((listOf(SUMMARY_HEADER) + rows).joinToString("\n") + "\n")

    private fun writeMeta(dir: File, startWall: Long, endWall: Long, seconds: Int) = File(dir, "meta.txt").writeText(
        "model=${Build.MODEL}\nbuild=${Build.DISPLAY}\nsdk_int=${Build.VERSION.SDK_INT}\n" +
            "start_wall_ms=$startWall\nend_wall_ms=$endWall\nseconds=$seconds\n"
    )

    private fun writeSensorList(file: File) {
        val header = "name\tvendor\ttype\tstring_type\tmin_delay_us\tmax_delay_us\tfifo_max\tfifo_reserved\treporting_mode\twake_up\tpower_ma\tresolution\tmax_range"
        val rows = sensorManager.getSensorList(Sensor.TYPE_ALL).map {
            listOf(
                it.name, it.vendor, it.type, it.stringType, it.minDelay, it.maxDelay, it.fifoMaxEventCount,
                it.fifoReservedEventCount, it.reportingMode, it.isWakeUpSensor, it.power, it.resolution, it.maximumRange,
            ).joinToString("\t")
        }
        file.writeText((listOf(header) + rows).joinToString("\n") + "\n")
        log("visible sensors: ${rows.size}")
    }

    private fun writePermissions(file: File) {
        val rows = (PERMISSIONS + "com.samsung.permission.SSENSOR").map {
            "$it\t${checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED}"
        }
        file.writeText((listOf("permission\tgranted") + rows).joinToString("\n") + "\n")
    }

    private fun newRunDir(kind: String): File {
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        return File(filesDir, "runs/$stamp-$kind").apply { mkdirs() }
    }

    /** The laptop server runs are uploaded to; set once with `--es server http://host:port`, empty to clear. */
    private var server: String?
        get() = getSharedPreferences("probe", Context.MODE_PRIVATE).getString("server", null)?.takeIf { it.isNotEmpty() }
        set(value) {
            getSharedPreferences("probe", Context.MODE_PRIVATE).edit().putString("server", value).apply()
        }

    private fun goForeground(text: String) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Recording", NotificationManager.IMPORTANCE_LOW))
        val notification = Notification.Builder(this, CHANNEL)
            .setContentTitle("Pulse Probe")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .build()
        startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH)
    }

    private fun log(text: String) = Log.i(TAG, text)

    private fun String?.list() = this?.split(',')?.map(String::trim)?.filter(String::isNotEmpty)?.takeIf { it.isNotEmpty() }

    companion object {
        const val TAG = "PulseProbe"
        const val EXTRA_CMD = "cmd"
        const val EXTRA_SENSORS = "sensors"
        const val EXTRA_FEATURES = "features"
        const val EXTRA_FEATURE = "feature"
        const val EXTRA_SECONDS = "seconds"
        const val EXTRA_SERVER = "server"
        private const val CHANNEL = "recording"
        private val RUN_COMMANDS = setOf("probe", "record", "session", "spot", "caps", "detect")
        private const val PROBE_SECONDS = 8
        private const val SESSION_SECONDS = 60
        const val SPOT_MAX_SECONDS = 30
        private const val SDK_TIMEOUT_MS = 10_000L
        private const val FLUSH_WAIT_MS = 3_000L
        private const val SETTLE_MS = 1_000L
        /** Covers a detection (10 s connect + 3 s policy) plus an upload. */
        private const val DETECT_WAKE_MS = 90_000L
        private const val SUMMARY_HEADER = "tier\talias\tavailable\tregistered\tevents\trate_hz\tvalues_per_event\tfirst_values\terror"

        private fun defaultSeconds(cmd: String) = when (cmd) {
            "probe" -> PROBE_SECONDS
            "session" -> SESSION_SECONDS
            "spot" -> SPOT_MAX_SECONDS
            else -> 30
        }

        val PERMISSIONS = buildList {
            add(if (Build.VERSION.SDK_INT >= 36) "android.permission.health.READ_HEART_RATE" else android.Manifest.permission.BODY_SENSORS)
            add(android.Manifest.permission.ACTIVITY_RECOGNITION)
            add("com.samsung.android.hardware.sensormanager.permission.READ_ADDITIONAL_HEALTH_DATA")
            // Android 16 splits health data into granular permissions; Health Platform checks these too.
            if (Build.VERSION.SDK_INT >= 36) {
                add(SKIN_PERMISSION)
                add(SPO2_PERMISSION)
            }
        }

        /** Hands an activity intent's command and extras to the service. */
        fun start(context: Context, intent: Intent) {
            if (intent.getStringExtra(EXTRA_CMD) == null) return
            context.startForegroundService(Intent(context, RecorderService::class.java).putExtras(intent))
        }
    }
}
