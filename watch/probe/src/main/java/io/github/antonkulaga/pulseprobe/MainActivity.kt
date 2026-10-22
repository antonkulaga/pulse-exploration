package io.github.antonkulaga.pulseprobe

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * Shows what this watch can record (tier and, per feature, the sources or why not) and hands commands to
 * [RecorderService], which does the work in the foreground so it keeps running when the screen goes to
 * ambient.
 *
 * From adb (all through this activity; the service is not exported):
 *   am start -n io.github.antonkulaga.pulseprobe/.MainActivity --es cmd caps
 *   am start -n io.github.antonkulaga.pulseprobe/.MainActivity --es cmd session --ei seconds 300
 *   am start -n io.github.antonkulaga.pulseprobe/.MainActivity --es cmd session --es features ppg,hr,ibi,motion,skin_temp --ei seconds 60
 *   am start -n io.github.antonkulaga.pulseprobe/.MainActivity --es cmd spot --es feature ecg
 *   am start -n io.github.antonkulaga.pulseprobe/.MainActivity --es cmd probe
 *   am start -n io.github.antonkulaga.pulseprobe/.MainActivity --es cmd record --es sensors hr_raw,hr,acc --ei seconds 60
 *   am start -n io.github.antonkulaga.pulseprobe/.MainActivity --es cmd wifi --ei seconds 1800
 *   add --es server http://<laptop>:8765 once; later runs upload there
 */
class MainActivity : Activity() {

    private var pendingIntent: Intent? = null
    private lateinit var tierView: TextView
    private lateinit var featureList: LinearLayout
    private lateinit var activityView: TextView
    /** Buttons that need a tier "all" feature: button, feature, label. */
    private val sdkButtons = mutableListOf<Triple<Button, String, String>>()
    /** Buttons whose command the service ignores while it is busy. */
    private val runButtons = mutableListOf<Button>()
    private val onChange: () -> Unit = { runOnUiThread { render() } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Round 480x480 screen: generous side and bottom padding keeps text inside the circle.
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(26), dp(30), dp(26), dp(70))
        }
        layout.addView(text(14f, Gravity.CENTER).apply {
            typeface = Typeface.DEFAULT_BOLD
            text = "Pulse Probe"
        })
        tierView = text(13f, Gravity.CENTER).also(layout::addView)
        featureList = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(4), 0, dp(4))
        }.also(layout::addView)
        activityView = text(12f, Gravity.CENTER).apply { setTextColor(ACCENT) }.also(layout::addView)

        layout.addView(button("Continuous 60 s") { send("session", seconds = 60) }.also(runButtons::add))
        layout.addView(button("Continuous 5 min") { send("session", seconds = 300) }.also(runButtons::add))
        layout.addView(sdkButton("PPG 100 Hz 30 s", "ppg100"))
        layout.addView(sdkButton("ECG 30 s (finger on the top key)", "ecg"))
        layout.addView(button("Keep Wi-Fi 30 min") { send("wifi", seconds = 1800) })
        layout.addView(button("Check watch again") { send("caps") }.also(runButtons::add))
        layout.addView(text(11f, Gravity.CENTER).apply {
            setTextColor(Color.GRAY)
            text = "${Build.MODEL}, API ${Build.VERSION.SDK_INT}\nadb logcat -s PulseProbe"
        })
        setContentView(ScrollView(this).apply { addView(layout) })

        pendingIntent = intent
        val missing = RecorderService.PERMISSIONS.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) runPending(recheck = false) else requestPermissions(missing.toTypedArray(), 1)
    }

    override fun onStart() {
        super.onStart()
        ProbeState.addListener(onChange)
        render()
    }

    override fun onStop() {
        ProbeState.removeListener(onChange)
        super.onStop()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        pendingIntent = intent
        runPending(recheck = false)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        // Granted permissions change what the SDK allows (heart rate, skin temperature, SpO2).
        runPending(recheck = true)
    }

    /** Runs a command that came with the intent; without one, makes sure the watch has been checked. */
    private fun runPending(recheck: Boolean) {
        val intent = pendingIntent
        pendingIntent = null
        if (intent?.getStringExtra(RecorderService.EXTRA_CMD) != null) return RecorderService.start(this, intent)
        if (recheck || ProbeState.capabilities == null) send("detect")
    }

    private fun render() {
        val caps = ProbeState.capabilities
        tierView.text = when {
            caps == null && ProbeState.detecting -> "Checking the watch…"
            caps == null -> "Not checked yet"
            caps.tier == TIER_ALL -> "Tier all: Samsung SDK + Android sensors"
            else -> "Tier 25hz: Android sensors only\n(${caps.sdk.reason})"
        }
        featureList.removeAllViews()
        caps?.features?.forEach { (name, f) ->
            featureList.addView(text(12f, Gravity.START).apply {
                text = if (f.available) "$name: ${f.tier}, ${f.sources.joinToString(" + ")}" else "$name: ${f.reason}"
                setTextColor(if (f.available) Color.WHITE else Color.GRAY)
            })
        }
        val idle = ProbeState.activity.isEmpty()
        activityView.text = ProbeState.activity
        activityView.visibility = if (idle || (caps == null && ProbeState.detecting)) View.GONE else View.VISIBLE
        runButtons.forEach { it.isEnabled = idle }
        for ((button, feature, label) in sdkButtons) {
            val f = caps?.features?.get(feature)
            button.isEnabled = idle && f?.available == true
            button.text = when {
                f == null -> label
                f.available -> label
                else -> "$label\n${f.reason}"
            }
        }
    }

    private fun send(cmd: String, seconds: Int? = null, feature: String? = null) =
        RecorderService.start(this, Intent().apply {
            putExtra(RecorderService.EXTRA_CMD, cmd)
            seconds?.let { putExtra(RecorderService.EXTRA_SECONDS, it) }
            feature?.let { putExtra(RecorderService.EXTRA_FEATURE, it) }
        })

    private fun sdkButton(label: String, feature: String) =
        button(label) { send("spot", feature = feature) }.also {
            it.isEnabled = false
            sdkButtons += Triple(it, feature, label)
        }

    private fun button(label: String, onClick: () -> Unit) = Button(this).apply {
        text = label
        isAllCaps = false
        textSize = 13f
        setOnClickListener { onClick() }
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
    }

    private fun text(size: Float, gravity: Int) = TextView(this).apply {
        textSize = size
        this.gravity = gravity
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private companion object {
        val ACCENT = Color.rgb(0x80, 0xCB, 0xC4)
    }
}
