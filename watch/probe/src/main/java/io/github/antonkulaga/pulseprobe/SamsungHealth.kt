package io.github.antonkulaga.pulseprobe

import android.content.Context
import android.os.Handler
import java.io.File

/** The package that hosts the Samsung Health Sensor SDK service on the watch. */
const val HEALTH_PLATFORM_PACKAGE = "com.samsung.android.service.health"

/**
 * One connection to Samsung's Health Platform, or a stand-in when the app was built without the SDK
 * .aar. `openSamsungHealth(context)` returns one; it is defined once in src/samsungSdk (the real SDK) and
 * once in src/noSamsungSdk (the stand-in), and build.gradle.kts compiles whichever fits.
 *
 * Nothing here throws: failures end up in [error] and [reason].
 */
interface SamsungHealth {
    /** Health Platform present on the watch and new enough for the SDK. */
    val installed: Boolean
    val connected: Boolean
    /** What went wrong, technically; empty when connected. */
    val error: String
    /** Why tier "all" is off at the connection level (a REASON_ constant); empty when connected. */
    val reason: String
    /** SDK capability version, such as 1.4.01.00. */
    val version: String
    /** HealthTrackerType names this watch supports. */
    val supported: List<String>

    /** Connects; [onResult] runs exactly once, on any thread: connected, failed, or [timeout] was called. */
    fun connect(onResult: () -> Unit)

    /** Gives up waiting for the Health Platform. */
    fun timeout()

    /**
     * Starts tracker [type] for up to [waitMs] to learn whether this app may use the SDK, then stops it.
     * [onResult] runs once, on [handler], with a POLICY_ constant and an error (empty unless something failed).
     */
    fun checkPolicy(type: String, waitMs: Long, handler: Handler, onResult: (policy: String, error: String) -> Unit)

    /** A tier "all" source recording [tracker] through this connection, writing [file] when given one. */
    fun source(tracker: SdkTracker, file: File?): Source

    fun disconnect()
}

/** Health Platform's versionName, or null when it is not installed (or not visible to this app). */
fun healthPlatformVersion(context: Context): String? = try {
    context.packageManager.getPackageInfo(HEALTH_PLATFORM_PACKAGE, 0).versionName.orEmpty()
} catch (_: Exception) {
    null
}
