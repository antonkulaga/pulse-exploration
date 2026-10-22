package io.github.antonkulaga.pulsephone

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Result of one upload attempt. [reachable] is false when the server could not be reached at all. */
class Outcome(val dir: File, val result: JSONObject?, val error: String?, val reachable: Boolean)

/**
 * One directory per recording under `<app files>/traces/<stamp>-<mode>/`:
 * - trace.json: exactly the body that is (or will be) POSTed,
 * - frames.csv: per-frame clipping and capture-result values,
 * - result.json: the server's answer; its absence marks the trace as pending a retry,
 * - error.txt: the last upload failure.
 */
class TraceStore(base: File) {
    val root = File(base, "traces")

    fun save(json: JSONObject, csv: String, mode: Mode): File {
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        var dir = File(root, "$stamp-${mode.wire}")
        var i = 1
        while (dir.exists()) dir = File(root, "$stamp-${mode.wire}-${i++}")
        dir.mkdirs()
        File(dir, TRACE).writeText(json.toString())
        File(dir, "frames.csv").writeText(csv)
        return dir
    }

    fun pending(): List<File> =
        root.listFiles()?.filter { File(it, TRACE).isFile && !File(it, RESULT).isFile }?.sortedBy { it.name }.orEmpty()

    /** Blocking; call off the main thread. */
    fun upload(dir: File, server: String): Outcome {
        val url = "${normalize(server)}/camera/trace"
        return try {
            val (code, text) = postJson(url, File(dir, TRACE).readBytes())
            if (code in 200..299) {
                val json = JSONObject(text)
                File(dir, RESULT).writeText(json.toString(1))
                File(dir, ERROR).delete()
                Outcome(dir, json, null, true)
            } else {
                val message = "HTTP $code from $url: ${detail(text)}"
                File(dir, ERROR).writeText("$message\n\n$text\n")
                Outcome(dir, null, message, true)
            }
        } catch (e: IOException) {
            val message = "server unreachable ($url): ${e.message ?: e}"
            File(dir, ERROR).writeText("$message\n")
            Outcome(dir, null, message, false)
        } catch (e: IllegalArgumentException) {
            val message = "bad server URL $server: ${e.message}"
            File(dir, ERROR).writeText("$message\n")
            Outcome(dir, null, message, false)
        } catch (e: JSONException) {
            val message = "server answer is not JSON: ${e.message}"
            File(dir, ERROR).writeText("$message\n")
            Outcome(dir, null, message, true)
        }
    }

    companion object {
        const val TRACE = "trace.json"
        const val RESULT = "result.json"
        const val ERROR = "error.txt"

        /** "192.168.1.177:8765/" -> "http://192.168.1.177:8765" */
        fun normalize(server: String): String = server.trim().trimEnd('/').let { if ("://" in it) it else "http://$it" }

        private fun postJson(url: String, body: ByteArray): Pair<Int, String> {
            val conn = URI.create(url).toURL().openConnection() as HttpURLConnection
            try {
                conn.requestMethod = "POST"
                conn.doOutput = true
                conn.connectTimeout = 5_000
                conn.readTimeout = 120_000 // analysis runs server-side and can take a while
                conn.setFixedLengthStreamingMode(body.size)
                conn.setRequestProperty("Content-Type", "application/json")
                conn.setRequestProperty("Accept", "application/json")
                conn.outputStream.use { it.write(body) }
                val code = conn.responseCode
                val stream = if (code >= 400) conn.errorStream else conn.inputStream
                return code to (stream?.bufferedReader()?.use { it.readText() } ?: "")
            } finally {
                conn.disconnect()
            }
        }

        /** FastAPI errors are {"detail": "..."} or {"detail": [{"msg": ...}, ...]}. */
        private fun detail(text: String): String = try {
            when (val d = JSONObject(text).opt("detail")) {
                is String -> d
                is JSONArray -> (0 until d.length()).joinToString("; ") { i ->
                    (d.opt(i) as? JSONObject)?.let { o -> o.optString("msg").ifEmpty { o.toString() } } ?: d.opt(i).toString()
                }
                null -> text.take(300)
                else -> d.toString()
            }
        } catch (_: JSONException) {
            text.take(300)
        }
    }
}
