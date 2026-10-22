package io.github.antonkulaga.pulseprobe

/*
 * What this watch can record, and through which source. Plain Kotlin (no Android imports) so the
 * resolution and the capabilities.json format can be checked on a laptop JVM.
 *
 * Tier "25hz": plain Android SensorManager, nothing for the user to set up.
 * Tier "all": the Samsung Health Sensor SDK is allowed (Health Platform developer mode on the watch, or
 * Samsung partner approval). Everything tier "25hz" has keeps working in tier "all".
 */

const val TIER_ALL = "all"
const val TIER_25HZ = "25hz"
const val POLICY_ALLOWED = "allowed"
const val POLICY_DENIED = "denied"
const val POLICY_UNKNOWN = "unknown"

const val SKIN_PERMISSION = "android.permission.health.READ_SKIN_TEMPERATURE"
const val SPO2_PERMISSION = "android.permission.health.READ_OXYGEN_SATURATION"
private val PERMISSION_NAMES = mapOf(SKIN_PERMISSION to "skin temperature", SPO2_PERMISSION to "blood oxygen")

/** Why the Samsung SDK features are off, shown on the watch and written as each feature's reason. */
const val REASON_NOT_BUILT = "app built without the Samsung Health Sensor SDK"
const val REASON_NOT_INSTALLED = "needs Samsung Health Platform (not installed)"
const val REASON_OLD_PLATFORM = "needs a newer Samsung Health Platform"
const val REASON_NOT_CONNECTED = "Samsung Health Platform did not connect"
const val REASON_DEVELOPER_MODE = "needs Samsung Health Platform developer mode"
const val REASON_HR_PERMISSION = "needs the heart-rate permission"
const val REASON_POLICY_UNKNOWN = "could not check Samsung SDK access"
const val REASON_UNSUPPORTED = "not supported on this watch"

/** A Samsung Health Sensor SDK tracker the app records; its CSV columns live with the SDK code. */
data class SdkTracker(val alias: String, val type: String, val onDemand: Boolean, val permission: String? = null)

val SDK_TRACKERS: Map<String, SdkTracker> = listOf(
    SdkTracker("sdk_ppg", "PPG_CONTINUOUS", false),
    SdkTracker("sdk_hr", "HEART_RATE_CONTINUOUS", false),
    SdkTracker("sdk_acc", "ACCELEROMETER_CONTINUOUS", false),
    SdkTracker("sdk_skin", "SKIN_TEMPERATURE_CONTINUOUS", false, SKIN_PERMISSION),
    SdkTracker("sdk_eda", "EDA_CONTINUOUS", false),
    SdkTracker("sdk_ppg100", "PPG_ON_DEMAND", true),
    SdkTracker("sdk_ecg", "ECG_ON_DEMAND", true),
    SdkTracker("sdk_spo2", "SPO2_ON_DEMAND", true, SPO2_PERMISSION),
).associateBy { it.alias }

/** Tier "25hz" sources features use, with the name used in reasons. */
val ANDROID_SOURCES = linkedMapOf(
    "hr_raw" to "Samsung raw PPG sensor",
    "hr" to "heart-rate sensor",
    "acc" to "accelerometer",
    "gyro" to "gyroscope",
)

/** Trackers tried, in order, to learn whether the SDK policy lets this app in. */
val POLICY_TYPES = listOf("HEART_RATE_CONTINUOUS", "PPG_CONTINUOUS", "ACCELEROMETER_CONTINUOUS")

/**
 * A signal the user asks for. [candidates] are its sources, preferred first; every available one is
 * recorded. A feature with [requires] is available only when that SDK source is.
 */
class FeatureSpec(val name: String, val candidates: List<String>, val requires: String? = null, val onDemand: Boolean = false)

val FEATURES: List<FeatureSpec> = listOf(
    FeatureSpec("ppg", listOf("sdk_ppg", "hr_raw")),
    FeatureSpec("hr", listOf("sdk_hr", "hr")),
    FeatureSpec("ibi", listOf("sdk_hr"), requires = "sdk_hr"),
    FeatureSpec("motion", listOf("acc", "gyro")),
    FeatureSpec("skin_temp", listOf("sdk_skin"), requires = "sdk_skin"),
    // On-demand: at most 30 s, and alone (continuous SDK trackers may be invalid meanwhile).
    FeatureSpec("ppg100", listOf("sdk_ppg100", "acc"), requires = "sdk_ppg100", onDemand = true),
    FeatureSpec("ecg", listOf("sdk_ecg"), requires = "sdk_ecg", onDemand = true),
    FeatureSpec("spo2", listOf("sdk_spo2"), requires = "sdk_spo2", onDemand = true),
)
val FEATURE_SPECS = FEATURES.associateBy { it.name }
val DEFAULT_SESSION_FEATURES = listOf("ppg", "hr", "ibi", "motion")

fun tierOfSource(alias: String) = if (alias in SDK_TRACKERS) TIER_ALL else TIER_25HZ

data class SdkStatus(
    val installed: Boolean,
    val connected: Boolean,
    val policy: String,
    val error: String,
    val version: String,
    val supported: List<String>,
    /** Why SDK sources are off; empty when the policy is "allowed". Not written to JSON (features carry it). */
    val reason: String,
)

/**
 * Turns what the Health Platform answered into an [SdkStatus]. [connectionReason] is set by the
 * connection when it could not connect (one of the REASON_ constants); [policyError] comes from the
 * policy check.
 */
fun sdkStatus(
    installed: Boolean,
    connected: Boolean,
    connectionError: String,
    connectionReason: String,
    version: String,
    supported: List<String>,
    policy: String,
    policyError: String,
): SdkStatus {
    val reason = when {
        policy == POLICY_ALLOWED -> ""
        !connected -> connectionReason.ifEmpty { REASON_NOT_CONNECTED }
        policy == POLICY_DENIED -> REASON_DEVELOPER_MODE
        "PERMISSION_ERROR" in policyError -> REASON_HR_PERMISSION
        else -> REASON_POLICY_UNKNOWN
    }
    val error = listOf(connectionError, policyError).filter(String::isNotEmpty).joinToString("; ")
    return SdkStatus(installed, connected, policy, error, version, supported, reason)
}

data class Feature(val available: Boolean, val tier: String?, val sources: List<String>, val reason: String)

data class SourceStatus(val available: Boolean, val reason: String)

class Capabilities(
    val sdk: SdkStatus,
    /** hr_raw, hr, acc, gyro: whether the watch has the sensor. */
    val android: Map<String, Boolean>,
    /** Runtime permissions SDK trackers need that are not granted. */
    val missingPermissions: Set<String>,
    val features: Map<String, Feature>,
) {
    val tier: String get() = if (sdk.policy == POLICY_ALLOWED) TIER_ALL else TIER_25HZ

    /** Whether [alias] can be recorded now, and why not. */
    fun source(alias: String): SourceStatus {
        val tracker = SDK_TRACKERS[alias]
        if (tracker == null) {
            val name = ANDROID_SOURCES[alias] ?: alias
            return if (android[alias] == true) SourceStatus(true, "") else SourceStatus(false, "no $name on this watch")
        }
        return when {
            sdk.policy != POLICY_ALLOWED -> SourceStatus(false, sdk.reason)
            tracker.type !in sdk.supported -> SourceStatus(false, REASON_UNSUPPORTED)
            tracker.permission != null && tracker.permission in missingPermissions ->
                SourceStatus(false, "needs the ${PERMISSION_NAMES[tracker.permission] ?: tracker.permission} permission")
            else -> SourceStatus(true, "")
        }
    }

    /**
     * This run's view after recording: a source that failed, or delivered nothing (so left no CSV), is
     * dropped from its features' sources unless it delivered data (a feature left with none becomes
     * unavailable), and every such source is noted in the feature's reason. [outcomes] maps each alias
     * opened to (events, error).
     */
    fun afterRun(outcomes: Map<String, Pair<Int, String>>): Capabilities {
        val updated = features.mapValues { (_, f) ->
            val failed = f.sources.filter { a -> outcomes[a]?.let { it.second.isNotEmpty() || it.first == 0 } == true }
            if (failed.isEmpty()) return@mapValues f
            val note = failed.joinToString("; ") { a ->
                val error = outcomes.getValue(a).second
                if (error.isNotEmpty()) "$a failed: $error" else "$a delivered no data"
            }
            val kept = f.sources.filter { it !in failed || outcomes.getValue(it).first > 0 }
            if (kept.isEmpty()) Feature(false, null, emptyList(), note)
            else Feature(true, tierOf(kept), kept, listOf(f.reason, note).filter(String::isNotEmpty).joinToString("; "))
        }
        return Capabilities(sdk, android, missingPermissions, updated)
    }

    /** A <kind>.tsv row for a source that was not opened (caps, or a refused spot). */
    fun unopenedRow(alias: String): String {
        val s = source(alias)
        return listOf(tierOfSource(alias), alias, s.available, false, 0, "", 0, "", s.reason).joinToString("\t")
    }

    /** Every source a feature can use, in feature order. */
    fun featureSources(): List<String> = FEATURES.flatMap { it.candidates }.distinct()

    fun describe(name: String): String {
        val f = features.getValue(name)
        return if (f.available) "$name: ${f.tier} via ${f.sources.joinToString(",")}" + (if (f.reason.isEmpty()) "" else " (${f.reason})")
        else "$name: unavailable, ${f.reason}"
    }

    fun toJson(): String = Json.write(
        linkedMapOf(
            "schema" to 1,
            "tier" to tier,
            "sdk" to linkedMapOf(
                "installed" to sdk.installed,
                "connected" to sdk.connected,
                "policy" to sdk.policy,
                "error" to sdk.error,
                "version" to sdk.version,
                "supported" to sdk.supported,
            ),
            "android" to linkedMapOf("hr_raw" to android["hr_raw"], "hr" to android["hr"], "acc" to android["acc"], "gyro" to android["gyro"])
                .mapValues { it.value == true },
            "features" to features.mapValues { (_, f) ->
                linkedMapOf("available" to f.available, "tier" to f.tier, "sources" to f.sources, "reason" to f.reason)
            },
        )
    ) + "\n"

    companion object {
        fun build(sdk: SdkStatus, android: Map<String, Boolean>, missingPermissions: Set<String>): Capabilities {
            val probe = Capabilities(sdk, android, missingPermissions, emptyMap())
            val features = FEATURES.associate { spec ->
                val status = spec.candidates.associateWith(probe::source)
                val blocked = spec.requires?.takeIf { status.getValue(it).available.not() }
                val sources = spec.candidates.filter { status.getValue(it).available }
                spec.name to when {
                    blocked != null -> Feature(false, null, emptyList(), status.getValue(blocked).reason)
                    sources.isEmpty() -> Feature(false, null, emptyList(), status.values.map { it.reason }.distinct().joinToString("; "))
                    else -> Feature(true, tierOf(sources), sources, "")
                }
            }
            return Capabilities(sdk, android, missingPermissions, features)
        }

        private fun tierOf(sources: List<String>) = if (sources.any { it in SDK_TRACKERS }) TIER_ALL else TIER_25HZ
    }
}

/** Just enough JSON: objects nested two levels deep or more go on one line, so each feature is one line. */
internal object Json {
    fun write(value: Any?, depth: Int = 0): String = when (value) {
        null -> "null"
        is String -> quote(value)
        is Boolean, is Number -> value.toString()
        is List<*> -> value.joinToString(", ", "[", "]") { write(it, depth + 1) }
        is Map<*, *> -> if (depth >= 2 || value.isEmpty()) {
            value.entries.joinToString(", ", "{", "}") { "${quote(it.key.toString())}: ${write(it.value, depth + 1)}" }
        } else {
            val pad = "  ".repeat(depth + 1)
            value.entries.joinToString(",\n", "{\n", "\n" + "  ".repeat(depth) + "}") {
                "$pad${quote(it.key.toString())}: ${write(it.value, depth + 1)}"
            }
        }
        else -> quote(value.toString())
    }

    private fun quote(s: String) = buildString {
        append('"')
        for (c in s) when {
            c == '"' -> append("\\\"")
            c == '\\' -> append("\\\\")
            c == '\n' -> append("\\n")
            c == '\t' -> append("\\t")
            c < ' ' -> append("\\u%04x".format(c.code))
            else -> append(c)
        }
        append('"')
    }
}
