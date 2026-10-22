package io.github.antonkulaga.pulseprobe

import android.content.Context
import android.os.Handler
import java.io.File

/**
 * Compiled when watch/probe/libs has no Samsung Health Sensor SDK .aar (or with -PsamsungSdk=false):
 * the app then records tier "25hz" only, and every tier "all" source reports why.
 */
fun openSamsungHealth(context: Context): SamsungHealth = NoSamsungSdk(context)

class NoSamsungSdk(context: Context) : SamsungHealth {
    override val installed = healthPlatformVersion(context) != null
    override val connected = false
    override val error = "$REASON_NOT_BUILT (no .aar in watch/probe/libs at build time)"
    override val reason = REASON_NOT_BUILT
    override val version = ""
    override val supported = emptyList<String>()

    override fun connect(onResult: () -> Unit) = onResult()
    override fun timeout() {}

    override fun checkPolicy(type: String, waitMs: Long, handler: Handler, onResult: (String, String) -> Unit) {
        handler.post { onResult(POLICY_UNKNOWN, "") }
    }

    override fun source(tracker: SdkTracker, file: File?): Source = UnavailableSource(tracker.alias, Tier.FULL, REASON_NOT_BUILT)

    override fun disconnect() {}
}
