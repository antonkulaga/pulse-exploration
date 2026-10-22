package io.github.antonkulaga.pulsephone

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

/**
 * Turns whatever the server returned into text. Every key is optional and may have another type than
 * expected (warnings as strings or as {code, severity, message}, quality as a label or an object,
 * waveform as a list or as {values: [...]}), so nothing here assumes a shape.
 */
object ResultFormatter {
    class Shown(val headline: String, val details: String, val waveform: FloatArray?)

    fun format(r: JSONObject): Shown {
        val hr = number(r, "hr_bpm")
        val lines = mutableListOf<String>()
        // A poor-quality number (no fingertip, motion, compression) must not look like a measurement.
        val poor = (r.opt("quality") as? JSONObject)?.optString("label") == "poor"
        if (poor && hr != null) lines += "NOT RELIABLE - see the warnings below"
        listOfNotNull(text(r, "tier")?.let { "tier $it" }, text(r, "mode"), quality(r.opt("quality")))
            .takeIf { it.isNotEmpty() }?.let { lines += it.joinToString(" · ") }
        number(r, "hr_peaks_bpm")?.let { lines += f("beat-to-beat %.1f bpm", it) }
        listOfNotNull(
            number(r, "fps_measured")?.let { f("%.1f fps", it) },
            number(r, "duration_s")?.let { f("%.0f s", it) },
            number(r, "n_samples")?.let { "${it.toLong()} samples" },
            number(r, "dropped_frames")?.let { "${it.toLong()} gaps" },
        ).takeIf { it.isNotEmpty() }?.let { lines += it.joinToString(" · ") }
        hrv(r)?.let { lines += it }
        breathing(r)?.let { lines += it }
        number(r, "clipped_fraction")?.let { lines += f("clipped %.1f %%", it * 100) }
        for (w in warnings(r.opt("warnings"))) lines += "warning: $w"
        text(r, "error")?.let { lines += "error: $it" }
        text(r, "id")?.let { lines += "server id $it" }
        val headline = hr?.let { if (poor) f("(%.0f bpm?)", it) else f("%.0f bpm", it) } ?: "-- bpm"
        return Shown(headline, lines.joinToString("\n"), waveform(r.opt("waveform")))
    }

    private fun f(pattern: String, vararg args: Any) = String.format(Locale.US, pattern, *args)

    private fun number(o: JSONObject, key: String): Double? = (o.opt(key) as? Number)?.toDouble()?.takeIf { it.isFinite() }

    private fun text(o: JSONObject, key: String): String? = when (val v = o.opt(key)) {
        is String -> v.takeIf { it.isNotBlank() }
        is Number, is Boolean -> v.toString()
        else -> null
    }

    private fun quality(v: Any?): String? = when (v) {
        is JSONObject -> listOfNotNull(
            text(v, "label")?.uppercase(Locale.US),
            number(v, "snr_db")?.let { f("SNR %.1f dB", it) },
            number(v, "template_corr")?.let { f("template %.2f", it) },
            number(v, "perfusion_index_pct")?.let { f("PI %.2f %%", it) },
        ).takeIf { it.isNotEmpty() }?.joinToString(", ", prefix = "quality ")
        is String -> "quality $v"
        is Number -> "quality $v"
        else -> null
    }

    private fun hrv(r: JSONObject): String? {
        val parts = when (val v = r.opt("hrv")) {
            is JSONObject -> v.keys().asSequence().mapNotNull { k -> number(v, k)?.let { f("%s %.1f", k, it) } }.take(6).toList()
            is Number -> listOf("$v")
            else -> emptyList()
        }
        val note = text(r, "hrv_note")
        if (parts.isEmpty() && note == null) return null
        if (parts.isEmpty()) return if (note!!.startsWith("HRV", ignoreCase = true)) note else "HRV: $note"
        val reliable = (r.opt("hrv_reliable") as? Boolean)?.let { if (it) " (reliable)" else " (exploratory)" } ?: ""
        return "HRV: " + parts.joinToString(", ") + reliable + (note?.let { "\n  $it" } ?: "")
    }

    private fun breathing(r: JSONObject): String? {
        number(r, "resp_rate")?.let { return f("breathing %.1f /min", it) }
        val resp = r.opt("respiration") as? JSONObject ?: return null
        val rate = number(resp, "rate_brpm") ?: return text(resp, "note")?.let { "breathing: $it" }
        val reliable = (resp.opt("reliable") as? Boolean)?.let { if (it) "" else " (unreliable)" } ?: ""
        return f("breathing %.1f /min", rate) + reliable
    }

    private fun warnings(v: Any?): List<String> = when (v) {
        is JSONArray -> (0 until v.length()).mapNotNull { i ->
            when (val w = v.opt(i)) {
                is String -> w
                is JSONObject -> w.optString("message").ifEmpty { w.optString("code") }.ifEmpty { w.toString() }
                null, JSONObject.NULL -> null
                else -> w.toString()
            }
        }
        is String -> listOf(v)
        else -> emptyList()
    }

    /** The plotted waveform, at most ~1500 points. */
    private fun waveform(v: Any?): FloatArray? {
        val values = when (v) {
            is JSONArray -> v
            is JSONObject -> v.opt("values") as? JSONArray
            else -> null
        } ?: return null
        val n = values.length()
        if (n < 2) return null
        val step = (n + 1499) / 1500
        val out = ArrayList<Float>(n / step + 1)
        var i = 0
        while (i < n) {
            (values.opt(i) as? Number)?.toFloat()?.takeIf { it.isFinite() }?.let(out::add)
            i += step
        }
        return if (out.size < 2) null else out.toFloatArray()
    }
}
