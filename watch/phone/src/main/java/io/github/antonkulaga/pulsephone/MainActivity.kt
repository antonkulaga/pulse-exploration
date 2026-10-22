package io.github.antonkulaga.pulsephone

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.InputType
import android.util.Log
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject
import java.io.IOException
import java.time.Instant
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Camera PPG: records per-frame ROI colour means (never video) from the fingertip (rear camera + torch)
 * or the face (front camera), then POSTs them to the laptop server's `/camera/trace`.
 *
 * A run: bind the camera, wait for the first frame, settle 2 s with AE/AWB free (torch on for the
 * finger), lock AE/AWB, record N seconds on the sensor clock, save trace.json locally, upload.
 *
 * From adb (grant the camera once with `adb install -g` or `pm grant <pkg> android.permission.CAMERA`):
 *   am start -n io.github.antonkulaga.pulsephone/.MainActivity --es mode finger --ei seconds 45 --ez autostart true
 *   optional: --es server http://<laptop>:8765 (persisted), --ez fps60 true,
 *             --ei exposure_us 8000 --ei iso 100 (manual exposure; --ez manual false goes back to auto)
 *   re-send traces the server never answered: --ez retry true
 *   results:  adb logcat -s PulsePhone
 */
class MainActivity : ComponentActivity() {

    private enum class State(val label: String) {
        IDLE("ready"), OPENING("opening camera"), SETTLING("settling"), LOCKING("locking exposure"),
        RECORDING("recording"), SENDING("sending"),
    }

    private lateinit var prefs: Prefs
    private lateinit var store: TraceStore
    private lateinit var camera: PpgCamera
    private val main = Handler(Looper.getMainLooper())
    private val analysisExecutor = Executors.newSingleThreadExecutor { Thread(it, "ppg-analysis") }
    private val ioExecutor = Executors.newSingleThreadExecutor { Thread(it, "ppg-io") }
    private val analyzer = FrameAnalyzer(::onFrame)
    private val fpsMeter = FpsMeter()

    private lateinit var previewView: PreviewView
    private lateinit var guide: GuideOverlay
    private lateinit var livePlot: SignalPlotView
    private lateinit var statusText: TextView
    private lateinit var modeGroup: RadioGroup
    private lateinit var secondsLabel: TextView
    private lateinit var secondsBar: SeekBar
    private lateinit var serverField: EditText
    private lateinit var fps60Box: CheckBox
    private lateinit var manualBox: CheckBox
    private lateinit var exposureField: EditText
    private lateinit var isoField: EditText
    private lateinit var cameraText: TextView
    private lateinit var startButton: Button
    private lateinit var retryButton: Button
    private lateinit var hrText: TextView
    private lateinit var resultText: TextView
    private lateinit var controlsScroll: ScrollView
    private lateinit var resultPlot: SignalPlotView

    private var state = State.IDLE
    /** Bumped whenever a run starts, ends or is aborted; delayed callbacks of an older run then do nothing. */
    private var runId = 0
    private var inRun = false
    private var run: RecordingConfig? = null
    private var settleStart = 0L
    private var torch = "not used"
    private var lockConfirmed = false
    private var lockLatencyMs = 0L
    private var lockSensorNs = 0L
    private var cameraInterrupted = false
    private var recordedUtc = ""
    private var note = ""
    private var autostart = false
    private var loadingFields = false
    @Volatile private var recorder: TraceRecorder? = null
    @Volatile private var lastFrame: Frame? = null
    @Volatile private var firstFrame: (() -> Unit)? = null
    @Volatile private var plotRed = true

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            ready()
        } else {
            autostart = false
            showError("Camera permission denied.")
        }
    }

    private val ticker = object : Runnable {
        override fun run() {
            updateStatus()
            main.postDelayed(this, 250)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = Prefs(this)
        store = TraceStore(getExternalFilesDir(null) ?: filesDir)
        autostart = applyExtras(intent) && savedInstanceState == null
        setContentView(buildUi())
        camera = PpgCamera(this, this, previewView, analyzer, analysisExecutor)
        camera.onInterrupted = ::onCameraInterrupted
        loadFields()
        applyMode()
        refreshPending()
        setState(State.IDLE)
        main.post(ticker)
        when {
            savedInstanceState == null && boolExtra(intent, EXTRA_RETRY) == true -> retryPending()
            hasCameraPermission() -> ready()
            else -> cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        val auto = applyExtras(intent)
        if (inRun || state == State.SENDING) {
            if (auto) log("busy (${state.label}), autostart ignored")
            return
        }
        loadFields()
        applyMode()
        autostart = auto
        when {
            state == State.OPENING -> {} // the idle bind in progress honours autostart when it is done
            boolExtra(intent, EXTRA_RETRY) == true -> retryPending()
            hasCameraPermission() -> ready()
            else -> cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onStop() {
        if (inRun) {
            if (state == State.RECORDING) finishRecording(early = true) else abort("Interrupted: the app left the foreground.")
        }
        if (::serverField.isInitialized) saveFields()
        super.onStop()
    }

    override fun onDestroy() {
        main.removeCallbacksAndMessages(null)
        analysisExecutor.shutdown()
        ioExecutor.shutdown()
        super.onDestroy()
    }

    private fun hasCameraPermission() = checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun ready() {
        if (autostart) {
            autostart = false
            startRecording()
        } else {
            openIdle()
        }
    }

    // ---------------------------------------------------------------------------------------- run

    /** Binds the camera for the current settings so the preview can be aimed; no recording. */
    private fun openIdle() {
        if (state != State.IDLE) return
        val cfg = readConfig()
        if (camera.isBound(cfg)) return
        val id = runId
        setState(State.OPENING)
        camera.open(cfg) { err ->
            if (id != runId || state != State.OPENING) return@open
            setState(State.IDLE)
            onCameraOpened(err)
            when {
                autostart -> ready()
                err == null && !camera.isBound(readConfig()) -> openIdle() // settings changed meanwhile
            }
        }
    }

    private fun startRecording() {
        if (state != State.IDLE) return log("busy (${state.label}), start ignored")
        if (!hasCameraPermission()) {
            autostart = true
            cameraPermission.launch(Manifest.permission.CAMERA)
            return
        }
        saveFields()
        val cfg = readConfig()
        val id = ++runId
        inRun = true
        run = cfg
        clearResult()
        setState(State.OPENING)
        log("start $cfg")
        if (camera.isBound(cfg)) {
            awaitFirstFrame(id, cfg)
        } else {
            camera.open(cfg) { err ->
                if (id != runId) return@open
                onCameraOpened(err)
                if (err != null) abort(err) else awaitFirstFrame(id, cfg)
            }
        }
    }

    /** Settling starts on the first frame; without one, 60 fps falls back to 30 once, else the run fails. */
    private fun awaitFirstFrame(id: Int, cfg: RecordingConfig) {
        // Both the timeout and the posted first-frame callback run on the main thread; whichever runs first
        // wins, so a frame landing just as the timeout fires cannot settle a camera that is being rebound.
        var handled = false
        val timeout = Runnable {
            if (id != runId || handled) return@Runnable
            handled = true
            firstFrame = null
            if (cfg.try60) {
                log("no frames at 60 fps within $FIRST_FRAME_TIMEOUT_MS ms, retrying at 30 fps")
                val fallback = cfg.copy(try60 = false)
                run = fallback
                camera.open(fallback) { err ->
                    if (id != runId) return@open
                    camera.applied.put("fps_60_fallback", "no frames at 60 fps")
                    onCameraOpened(err)
                    if (err != null) abort(err) else awaitFirstFrame(id, fallback)
                }
            } else {
                abort("No camera frames within ${FIRST_FRAME_TIMEOUT_MS / 1000} s.")
            }
        }
        firstFrame = {
            main.post {
                if (id == runId && !handled) {
                    handled = true
                    main.removeCallbacks(timeout)
                    settle(id, cfg)
                }
            }
        }
        main.postDelayed(timeout, FIRST_FRAME_TIMEOUT_MS)
    }

    private fun settle(id: Int, cfg: RecordingConfig) {
        setState(State.SETTLING)
        settleStart = SystemClock.elapsedRealtime()
        camera.unlock()
        torch = if (cfg.mode == Mode.FINGER) camera.setTorch(true) else "not used"
        main.postDelayed({ if (id == runId) lock(id, cfg) }, SETTLE_MS)
    }

    private fun lock(id: Int, cfg: RecordingConfig) {
        setState(State.LOCKING)
        val lockAe = cfg.manual == null || !camera.manualSupported
        camera.lock(lockAe) { confirmed, latencyMs, lockedAtNs ->
            if (id == runId) startTrace(id, cfg, confirmed, latencyMs, lockedAtNs)
        }
    }

    private fun startTrace(id: Int, cfg: RecordingConfig, confirmed: Boolean, latencyMs: Long, lockedAtNs: Long) {
        lockConfirmed = confirmed
        lockLatencyMs = latencyMs
        lockSensorNs = lockedAtNs
        cameraInterrupted = false
        if (!confirmed) log("AE/AWB lock not confirmed by capture results within the timeout; recording anyway")
        recordedUtc = Instant.now().toString()
        camera.captures.begin()
        recorder = TraceRecorder(cfg.seconds, lockedAtNs)
        setState(State.RECORDING)
        // Duration is counted on frames; this only fires if frames stop arriving.
        main.postDelayed({ if (id == runId) finishRecording(early = true) }, (cfg.seconds + WATCHDOG_EXTRA_S) * 1000L)
    }

    /** Analysis thread. */
    private fun onFrame(f: Frame) {
        lastFrame = f
        fpsMeter.add(f.tNs)
        livePlot.add((if (plotRed) f.r else f.g).toFloat())
        firstFrame?.let {
            firstFrame = null
            it()
        }
        if (recorder?.offer(f) == true) main.post { finishRecording(early = false) }
    }

    private fun finishRecording(early: Boolean) {
        if (state != State.RECORDING) return
        val rec = recorder ?: return
        val cfg = run ?: return
        recorder = null
        runId++
        inRun = false
        val frames = rec.stop()
        val samples = camera.captures.end()
        val echo = camera.captures.echo() // before unlocking, so it shows the locked state
        camera.setTorch(false)
        camera.unlock()
        if (frames.size < MIN_FRAMES) {
            setState(State.IDLE)
            showError("Only ${frames.size} frames recorded; nothing sent.")
            return
        }
        val applied = JSONObject(camera.applied.toString()).apply {
            put("torch", torch)
            put("settle_ms", SETTLE_MS)
            put("lock_confirmed", lockConfirmed)
            put("lock_latency_ms", lockLatencyMs)
            put("lock_sensor_ns", lockSensorNs)
            put("camera_interrupted", cameraInterrupted)
        }
        val ctx = TraceContext(
            cfg, describeApp(this), describeDevice(), camera.report, applied, echo, analyzer.geometry,
            analyzer.roiInCrop, early, recordedUtc,
        )
        val server = prefs.server
        note = "to $server"
        setState(State.SENDING)
        ioExecutor.execute {
            // Off the main thread: thousands of frames of JSON and CSV formatting would stall the UI.
            val built = TraceJson.build(frames, samples, ctx)
            val dir = try {
                store.save(built.json, built.csv, cfg.mode)
            } catch (e: IOException) {
                main.post {
                    setState(State.IDLE)
                    showError("Could not save the trace: $e")
                }
                return@execute
            }
            log("saved ${dir.absolutePath}: ${built.summary}")
            val outcome = store.upload(dir, server)
            main.post {
                setState(State.IDLE)
                showOutcome(outcome)
                refreshPending()
            }
        }
    }

    /**
     * The camera closed under a run (another app, a HAL error). It comes back with the torch off and
     * possibly the lock re-applied without settling, so the frames after it would not be comparable: a
     * recording is cut and sent as stopped early, a run still settling is aborted.
     */
    private fun onCameraInterrupted(type: androidx.camera.core.CameraState.Type) {
        if (!inRun) return
        when (state) {
            State.RECORDING -> {
                log("camera interrupted ($type) while recording; sending what was recorded")
                cameraInterrupted = true
                finishRecording(early = true)
            }
            State.SETTLING, State.LOCKING -> abort("Camera interrupted ($type); run aborted, start again.")
            else -> {} // still waiting for the first frame: its timeout covers this
        }
    }

    private fun abort(message: String) {
        runId++
        inRun = false
        firstFrame = null
        recorder = null
        camera.captures.end()
        camera.setTorch(false)
        camera.unlock()
        setState(State.IDLE)
        showError(message)
    }

    private fun retryPending() {
        if (state != State.IDLE) return
        saveFields()
        val dirs = store.pending()
        if (dirs.isEmpty()) return refreshPending()
        val server = prefs.server
        note = "${dirs.size} pending to $server"
        setState(State.SENDING)
        ioExecutor.execute {
            var last: Outcome? = null
            for (d in dirs) {
                val o = store.upload(d, server)
                log("retry ${d.name}: ${if (o.result != null) "ok" else o.error}")
                last = o
                if (!o.reachable) break
            }
            main.post {
                setState(State.IDLE)
                last?.let(::showOutcome)
                refreshPending()
                if (hasCameraPermission()) openIdle() // a retry at launch skipped the camera
            }
        }
    }

    // ----------------------------------------------------------------------------------- settings

    /** Extras write through to the persisted settings. Returns whether to start recording right away. */
    private fun applyExtras(intent: Intent?): Boolean {
        val i = intent ?: return false
        i.getStringExtra(EXTRA_MODE)?.let { s ->
            Mode.parse(s)?.let { prefs.mode = it } ?: log("unknown mode '$s', keeping ${prefs.mode.wire}")
        }
        intExtra(i, EXTRA_SECONDS)?.let { prefs.seconds = it }
        i.getStringExtra(EXTRA_SERVER)?.takeIf { it.isNotBlank() }?.let { prefs.server = it }
        boolExtra(i, EXTRA_FPS60)?.let { prefs.try60 = it }
        intExtra(i, EXTRA_EXPOSURE_US)?.let {
            prefs.exposureUs = it
            prefs.manual = true
        }
        intExtra(i, EXTRA_ISO)?.let {
            prefs.iso = it
            prefs.manual = true
        }
        boolExtra(i, EXTRA_MANUAL)?.let { prefs.manual = it }
        return boolExtra(i, EXTRA_AUTOSTART) == true
    }

    /** Accepts `--ei key N` and `--es key N`. */
    private fun intExtra(i: Intent, key: String): Int? {
        if (!i.hasExtra(key)) return null
        return i.getIntExtra(key, Int.MIN_VALUE).takeIf { it != Int.MIN_VALUE } ?: i.getStringExtra(key)?.trim()?.toIntOrNull()
    }

    /** Accepts `--ez key true` and `--es key true`. */
    private fun boolExtra(i: Intent, key: String): Boolean? {
        if (!i.hasExtra(key)) return null
        return i.getStringExtra(key)?.trim()?.lowercase()?.toBooleanStrictOrNull() ?: i.getBooleanExtra(key, false)
    }

    private fun readConfig() = RecordingConfig(
        mode = prefs.mode,
        seconds = prefs.seconds,
        try60 = prefs.try60,
        manual = if (prefs.manual) ManualExposure(prefs.exposureUs * 1_000L, prefs.iso) else null,
    )

    private fun loadFields() {
        loadingFields = true
        modeGroup.check(modeGroup.findViewWithTag<RadioButton>(prefs.mode).id)
        secondsBar.progress = prefs.seconds
        secondsLabel.text = getString(R.string.duration_label, prefs.seconds)
        serverField.setText(prefs.server)
        fps60Box.isChecked = prefs.try60
        manualBox.isChecked = prefs.manual
        exposureField.setText(prefs.exposureUs.toString())
        isoField.setText(prefs.iso.toString())
        loadingFields = false
    }

    private fun saveFields() {
        prefs.server = serverField.text.toString().ifBlank { Prefs.DEFAULT_SERVER }
        prefs.seconds = secondsBar.progress
        prefs.try60 = fps60Box.isChecked
        prefs.manual = manualBox.isChecked
        exposureField.text.toString().toIntOrNull()?.let { prefs.exposureUs = it }
        isoField.text.toString().toIntOrNull()?.let { prefs.iso = it }
    }

    private fun applyMode() {
        val finger = prefs.mode == Mode.FINGER
        analyzer.roi = if (finger) Roi.FINGER else Roi.FACE
        analyzer.roiInCrop = !finger
        plotRed = finger
        guide.show(
            if (finger) Roi.FINGER else Roi.FACE,
            if (finger) "Cover lens and flash with a fingertip" else "Forehead and cheeks inside the box",
        )
        livePlot.title = if (finger) "red mean (ROI)" else "green mean (ROI)"
        livePlot.color = if (finger) Color.rgb(255, 82, 82) else Color.rgb(105, 240, 174)
        livePlot.clear()
    }

    // ------------------------------------------------------------------------------------------ UI

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun buildUi(): View {
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        guide = GuideOverlay(this)
        val previewFrame = FrameLayout(this).apply {
            addView(previewView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
            addView(guide, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        }
        livePlot = SignalPlotView(this)
        statusText = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 12f
            setPadding(dp(10), dp(4), dp(10), dp(4))
        }
        modeGroup = RadioGroup(this).apply {
            orientation = RadioGroup.HORIZONTAL
            for (m in Mode.entries) addView(RadioButton(context).apply {
                id = View.generateViewId()
                text = m.label
                tag = m
            })
            setOnCheckedChangeListener { group, checkedId ->
                if (loadingFields || checkedId == View.NO_ID) return@setOnCheckedChangeListener
                prefs.mode = group.findViewById<RadioButton>(checkedId).tag as Mode
                applyMode()
                openIdle()
            }
        }
        secondsLabel = TextView(this)
        secondsBar = SeekBar(this).apply {
            min = Prefs.MIN_SECONDS
            max = Prefs.MAX_SECONDS
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(bar: SeekBar, value: Int, fromUser: Boolean) {
                    secondsLabel.text = getString(R.string.duration_label, value)
                }

                override fun onStartTrackingTouch(bar: SeekBar) {}
                override fun onStopTrackingTouch(bar: SeekBar) {
                    prefs.seconds = bar.progress
                }
            })
        }
        serverField = EditText(this).apply {
            isSingleLine = true
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            hint = Prefs.DEFAULT_SERVER
        }
        fps60Box = CheckBox(this).apply { text = getString(R.string.try_60) }
        manualBox = CheckBox(this).apply {
            text = getString(R.string.manual_exposure)
            setOnCheckedChangeListener { _, _ -> refreshControls() }
        }
        exposureField = numberField("exposure, µs")
        isoField = numberField("ISO")
        val manualRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(exposureField, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(isoField, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        }
        cameraText = TextView(this).apply { textSize = 11f }
        startButton = Button(this).apply {
            setOnClickListener {
                when (state) {
                    State.IDLE -> startRecording()
                    State.RECORDING -> finishRecording(early = true)
                    State.OPENING, State.SETTLING, State.LOCKING -> if (inRun) abort("Cancelled.")
                    State.SENDING -> {}
                }
            }
        }
        retryButton = Button(this).apply { setOnClickListener { retryPending() } }
        hrText = TextView(this).apply {
            textSize = 44f
            typeface = Typeface.DEFAULT_BOLD
        }
        resultText = TextView(this).apply {
            textSize = 13f
            setTextIsSelectable(true)
        }
        resultPlot = SignalPlotView(this).apply {
            title = "server waveform"
            color = Color.rgb(130, 177, 255)
            visibility = View.GONE
        }

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(4), dp(12), dp(24))
            // The latest result comes first so it is visible right after a run without scrolling.
            addView(hrText)
            addView(resultText)
            addView(resultPlot, LinearLayout.LayoutParams(MATCH_PARENT, dp(110)))
            addView(retryButton)
            addView(modeGroup)
            addView(secondsLabel)
            addView(secondsBar)
            addView(TextView(context).apply { text = getString(R.string.server_label) })
            addView(serverField)
            addView(fps60Box)
            addView(manualBox)
            addView(TextView(context).apply { text = getString(R.string.manual_fields) })
            addView(manualRow)
            addView(cameraText)
        }
        controlsScroll = ScrollView(this).apply { addView(controls) }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            keepScreenOn = true
            addView(previewFrame, LinearLayout.LayoutParams(MATCH_PARENT, (resources.displayMetrics.heightPixels * 0.34f).toInt()))
            addView(livePlot, LinearLayout.LayoutParams(MATCH_PARENT, dp(90)))
            addView(statusText, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
            // START stays pinned outside the scroll area so it is never pushed off screen.
            addView(startButton, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply {
                setMargins(dp(12), dp(4), dp(12), dp(4))
            })
            addView(controlsScroll, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
        }
        // targetSdk 36 is edge-to-edge: keep content clear of the status/navigation bars and the keyboard.
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            WindowInsetsCompat.CONSUMED
        }
        return root
    }

    private fun numberField(hintText: String) = EditText(this).apply {
        isSingleLine = true
        inputType = InputType.TYPE_CLASS_NUMBER
        hint = hintText
    }

    private fun setState(s: State) {
        state = s
        refreshControls()
        updateStatus()
    }

    private fun refreshControls() {
        val idle = state == State.IDLE
        for (i in 0 until modeGroup.childCount) modeGroup.getChildAt(i).isEnabled = idle
        secondsBar.isEnabled = idle
        serverField.isEnabled = idle
        fps60Box.isEnabled = idle
        manualBox.isEnabled = idle && camera.manualSupported
        exposureField.isEnabled = manualBox.isEnabled && manualBox.isChecked
        isoField.isEnabled = exposureField.isEnabled
        retryButton.isEnabled = idle
        startButton.isEnabled = idle || inRun
        startButton.text = when {
            idle -> getString(R.string.start)
            state == State.RECORDING -> getString(R.string.stop_and_send)
            inRun -> getString(R.string.cancel)
            state == State.SENDING -> getString(R.string.sending)
            else -> state.label
        }
    }

    private fun onCameraOpened(err: String?) {
        if (err != null) {
            cameraText.text = err
            log("camera: $err")
            return
        }
        val r = camera.report
        val manual = if (camera.manualSupported) {
            val e = camera.exposureRangeNs
            val iso = camera.isoRange
            String.format(Locale.US, "manual: exposure %.3f-%.0f ms, ISO %s-%s",
                (e?.lower ?: 0L) / 1e6, (e?.upper ?: 0L) / 1e6, iso?.lower ?: "?", iso?.upper ?: "?")
        } else {
            "manual exposure: not offered (no MANUAL_SENSOR)"
        }
        cameraText.text = "camera ${r.optString("camera_id")} ${r.optString("lens_facing")} · ${r.optString("hardware_level")}" +
            " · fps ranges ${r.optJSONArray("fps_ranges")}\n" +
            "applied: fps ${camera.applied.opt("fps_range")}, NR ${camera.applied.optString("noise_reduction")}, " +
            "edge ${camera.applied.optString("edge")}, stab ${camera.applied.optString("video_stabilization")}, " +
            "OIS ${camera.applied.optString("optical_stabilization")}\n$manual"
        refreshControls()
    }

    private fun updateStatus() {
        val cfg = run
        val head = when (state) {
            State.SETTLING -> String.format(Locale.US, "settling %.1f s", (SETTLE_MS - (SystemClock.elapsedRealtime() - settleStart)).coerceAtLeast(0L) / 1000.0)
            State.RECORDING -> recorder?.let {
                String.format(Locale.US, "recording %.0f s left (%d frames)", ((cfg?.seconds ?: 0) - it.elapsedNs / 1e9).coerceAtLeast(0.0), it.count)
            } ?: state.label
            State.SENDING -> "sending $note"
            else -> state.label
        }
        val f = lastFrame
        val frame = f?.let {
            String.format(Locale.US, "%4.1f fps  R %5.1f  G %5.1f  B %5.1f  clip %4.1f%%", fpsMeter.fps(), it.r, it.g, it.b, it.clipR * 100)
        } ?: "no frames yet"
        val c = camera.captures.latest
        val capture = c?.let {
            String.format(Locale.US, "AE %s  AWB %s  exp %.2f ms  ISO %s",
                Names.aeState(it.aeState), Names.awbState(it.awbState), (it.exposureNs ?: 0L) / 1e6, it.iso?.toString() ?: "?")
        } ?: ""
        statusText.text = listOf(head, frame, capture).joinToString("\n")
    }

    private fun clearResult() {
        hrText.text = ""
        resultText.text = ""
        resultPlot.show(null)
        resultPlot.visibility = View.GONE
    }

    private fun showOutcome(o: Outcome) {
        val result = o.result
        if (result != null) {
            val shown = ResultFormatter.format(result)
            hrText.text = shown.headline
            resultText.text = shown.details + "\nsaved as ${o.dir.name}"
            resultPlot.show(shown.waveform)
            resultPlot.visibility = if (shown.waveform != null) View.VISIBLE else View.GONE
            log("result ${o.dir.name}: ${shown.headline} | ${shown.details.replace('\n', '|')}")
        } else {
            hrText.text = getString(R.string.no_hr)
            resultText.text = getString(R.string.upload_failed, o.error ?: "unknown error", o.dir.name)
            log("upload failed for ${o.dir.name}: ${o.error}")
        }
        controlsScroll.post { controlsScroll.smoothScrollTo(0, 0) }
    }

    private fun showError(message: String) {
        resultText.text = message
        log(message)
        updateStatus()
    }

    private fun refreshPending() {
        val n = store.pending().size
        retryButton.text = getString(R.string.retry, n)
        retryButton.visibility = if (n > 0) View.VISIBLE else View.GONE
    }

    private fun log(text: String) {
        Log.i(PpgCamera.TAG, text)
    }

    private companion object {
        const val EXTRA_MODE = "mode"
        const val EXTRA_SECONDS = "seconds"
        const val EXTRA_SERVER = "server"
        const val EXTRA_AUTOSTART = "autostart"
        const val EXTRA_FPS60 = "fps60"
        const val EXTRA_MANUAL = "manual"
        const val EXTRA_EXPOSURE_US = "exposure_us"
        const val EXTRA_ISO = "iso"
        const val EXTRA_RETRY = "retry"
        const val SETTLE_MS = 2_000L
        const val FIRST_FRAME_TIMEOUT_MS = 5_000L
        const val WATCHDOG_EXTRA_S = 5
        const val MIN_FRAMES = 10
    }
}
