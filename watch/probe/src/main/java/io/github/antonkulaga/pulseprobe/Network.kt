package io.github.antonkulaga.pulseprobe

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.util.Log
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Keeps the watch on Wi-Fi. Wear OS drops Wi-Fi whenever it can route traffic over the phone's
 * Bluetooth, which also turns wireless adb off. An app that requests a Wi-Fi network keeps the radio
 * up for as long as it holds the request.
 */
class WifiHold(context: Context) {
    private val cm = context.getSystemService(ConnectivityManager::class.java)
    @Volatile var network: Network? = null
        private set
    private var callback: ConnectivityManager.NetworkCallback? = null

    @Synchronized
    fun acquire() {
        if (callback != null) return
        val cb = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(n: Network) {
                network = n
                Log.i(RecorderService.TAG, "wifi available: $n")
            }

            override fun onLost(n: Network) {
                if (network == n) network = null
                Log.i(RecorderService.TAG, "wifi lost: $n")
            }
        }
        val request = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .build()
        cm.requestNetwork(request, cb)
        callback = cb
    }

    @Synchronized
    fun release() {
        callback?.let { runCatching { cm.unregisterNetworkCallback(it) } }
        callback = null
        network = null
    }

    val held get() = callback != null
}

/**
 * Sends every file of a run to the laptop server: `PUT {base}/runs/{run}/{file}` with the raw bytes.
 * Blocking; call it off the main thread. Uses the held Wi-Fi network when there is one.
 */
object Uploader {
    fun upload(base: String, runDir: File, network: Network?): String {
        val files = runDir.listFiles()?.filter { it.isFile }?.sortedBy { it.name }.orEmpty()
        var ok = 0
        val failures = mutableListOf<String>()
        for (file in files) {
            val url = URL("${base.trimEnd('/')}/runs/${enc(runDir.name)}/${enc(file.name)}")
            try {
                val conn = (network?.openConnection(url) ?: url.openConnection()) as HttpURLConnection
                conn.requestMethod = "PUT"
                conn.doOutput = true
                conn.connectTimeout = 10_000
                conn.readTimeout = 30_000
                conn.setFixedLengthStreamingMode(file.length())
                conn.setRequestProperty("Content-Type", "application/octet-stream")
                file.inputStream().use { input -> conn.outputStream.use { input.copyTo(it) } }
                val code = conn.responseCode
                conn.disconnect()
                if (code in 200..299) ok++ else failures += "${file.name}: HTTP $code"
            } catch (e: Exception) {
                failures += "${file.name}: $e"
            }
        }
        return "uploaded $ok/${files.size} to $base" + if (failures.isEmpty()) "" else "; failed: ${failures.joinToString("; ")}"
    }

    private fun enc(s: String) = URLEncoder.encode(s, "UTF-8").replace("+", "%20")
}
