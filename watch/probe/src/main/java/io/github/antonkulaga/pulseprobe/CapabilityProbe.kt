package io.github.antonkulaga.pulseprobe

import android.content.Context
import android.content.pm.PackageManager
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.CopyOnWriteArrayList

/**
 * What the activity shows: the last detection and what the service is doing. Display only; the service
 * keeps its own cache for its lifetime.
 */
object ProbeState {
    @Volatile var capabilities: Capabilities? = null
        private set
    @Volatile var detecting = false
        private set
    /** What the service is doing right now, empty when idle. */
    @Volatile var activity = ""
        private set
    private val listeners = CopyOnWriteArrayList<() -> Unit>()

    fun addListener(l: () -> Unit) = listeners.add(l)
    fun removeListener(l: () -> Unit) = listeners.remove(l)

    fun detecting(on: Boolean) = update { detecting = on }
    fun publish(c: Capabilities) = update { capabilities = c; detecting = false }
    fun activity(text: String) = update { activity = text }

    private fun update(change: () -> Unit) {
        change()
        listeners.forEach { it() }
    }
}

/**
 * Finds out what this watch can record:
 *
 * 1. Android sensors: hr_raw, hr, acc, gyro present or not (instant).
 * 2. Samsung SDK: connect to the Health Platform, waiting at most [CONNECT_TIMEOUT_MS]. A missing or old
 *    platform answers at once (PACKAGE_NOT_INSTALLED / OLD_PLATFORM_VERSION); a working one in well under
 *    a second.
 * 3. Policy: start HEART_RATE_CONTINUOUS for up to [POLICY_WAIT_MS]. SDK_POLICY_ERROR (about 1 s after
 *    setEventListener when developer mode is off) means "denied"; data, or no error in time, "allowed";
 *    PERMISSION_ERROR is a permission problem and leaves the policy "unknown". The tracker is then unset
 *    and the connection closed.
 *
 * Everything runs on [handler]; nothing blocks it.
 */
class CapabilityProbe(private val context: Context, private val sensorManager: SensorManager, private val handler: Handler) {

    fun detect(onDone: (Capabilities) -> Unit) {
        val t0 = SystemClock.elapsedRealtime()
        ProbeState.detecting(true)
        val android = ANDROID_SOURCES.keys.associateWith { alias ->
            Catalog.android[alias]?.let { (stringType, type, _) -> findSensor(sensorManager, stringType, type) != null } ?: false
        }
        val missing = SDK_TRACKERS.values.mapNotNull { it.permission }.filterNot(::granted).toSet()
        val sdk = try {
            openSamsungHealth(context)
        } catch (e: Throwable) {
            if (e is VirtualMachineError) throw e
            Log.w(RecorderService.TAG, "openSamsungHealth failed", e)
            null
        }
        var connectMs = -1L
        var finished = false
        val timeout = Runnable { sdk?.timeout() }

        fun finish(policy: String, policyError: String) {
            if (finished) return
            finished = true
            handler.removeCallbacks(timeout)
            val status = if (sdk == null) {
                sdkStatus(false, false, "cannot load the Samsung SDK", REASON_NOT_CONNECTED, "", emptyList(), POLICY_UNKNOWN, "")
            } else {
                sdkStatus(sdk.installed, sdk.connected, sdk.error, sdk.reason, sdk.version, sdk.supported, policy, policyError)
            }
            sdk?.disconnect()
            val caps = Capabilities.build(status, android, missing)
            Log.i(
                RecorderService.TAG,
                "capabilities: tier=${caps.tier} installed=${status.installed} connected=${status.connected} " +
                    "policy=${status.policy} connect_ms=$connectMs total_ms=${SystemClock.elapsedRealtime() - t0} error=${status.error}",
            )
            ProbeState.publish(caps)
            onDone(caps)
        }

        if (sdk == null) return finish(POLICY_UNKNOWN, "")
        sdk.connect {
            handler.post {
                if (finished) return@post
                handler.removeCallbacks(timeout)
                connectMs = SystemClock.elapsedRealtime() - t0
                if (!sdk.connected) return@post finish(POLICY_UNKNOWN, "")
                val type = POLICY_TYPES.firstOrNull { it in sdk.supported }
                    ?: return@post finish(POLICY_UNKNOWN, "no continuous tracker to check the policy with")
                sdk.checkPolicy(type, POLICY_WAIT_MS, handler) { policy, error -> finish(policy, error) }
            }
        }
        handler.postDelayed(timeout, CONNECT_TIMEOUT_MS)
    }

    /** On Android 16 the SDK checks granular health permissions; before that BODY_SENSORS covers them. */
    private fun granted(permission: String): Boolean {
        val effective = if (Build.VERSION.SDK_INT >= 36) permission else android.Manifest.permission.BODY_SENSORS
        return context.checkSelfPermission(effective) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        const val CONNECT_TIMEOUT_MS = 10_000L
        const val POLICY_WAIT_MS = 3_000L
    }
}
