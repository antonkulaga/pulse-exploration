package io.github.antonkulaga.pulsephone

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.TypedValue
import android.view.View
import java.util.Locale

private fun View.sp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, v, resources.displayMetrics)

/**
 * Auto-scaled line plot. Live mode: [add] from any thread, the newest [capacity] samples scroll from the
 * right. Static mode: [show] a whole series (the server's waveform).
 */
class SignalPlotView(context: Context, private val capacity: Int = 360) : View(context) {
    private val ring = FloatArray(capacity)
    private var count = 0
    private var head = 0
    private val snapshot = FloatArray(capacity)
    @Volatile private var fixed: FloatArray? = null
    private val path = Path()
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        strokeJoin = Paint.Join.ROUND
    }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.LTGRAY; textSize = sp(12f) }

    var title = ""
        set(value) {
            field = value
            postInvalidate()
        }
    var color: Int = Color.RED
        set(value) {
            field = value
            postInvalidate()
        }

    fun add(v: Float) {
        synchronized(ring) {
            ring[head] = v
            head = (head + 1) % capacity
            if (count < capacity) count++
        }
        postInvalidateOnAnimation()
    }

    fun clear() {
        synchronized(ring) {
            count = 0
            head = 0
        }
        fixed = null
        postInvalidate()
    }

    fun show(values: FloatArray?) {
        fixed = values
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(BACKGROUND)
        canvas.drawText(title, sp(6f), sp(14f), label)
        val static = fixed
        var n: Int
        val data: FloatArray
        if (static != null) {
            data = static
            n = static.size
        } else {
            data = snapshot
            synchronized(ring) {
                n = count
                for (i in 0 until n) snapshot[i] = ring[(head - n + i + capacity) % capacity]
            }
        }
        if (n < 2) return
        var lo = data[0]
        var hi = data[0]
        for (i in 1 until n) {
            if (data[i] < lo) lo = data[i]
            if (data[i] > hi) hi = data[i]
        }
        val span = (hi - lo).coerceAtLeast(1e-3f)
        val top = sp(20f)
        val h = height - top - sp(4f)
        val slots = (if (static != null) n else capacity) - 1
        val dx = width.toFloat() / slots
        val x0 = width - (n - 1) * dx // live data enters from the right
        path.reset()
        for (i in 0 until n) {
            val x = x0 + i * dx
            val y = top + h * (1 - (data[i] - lo) / span)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        line.color = color
        canvas.drawPath(path, line)
        val range = String.format(Locale.US, "%.2f .. %.2f", lo, hi)
        canvas.drawText(range, width - label.measureText(range) - sp(6f), sp(14f), label)
    }

    private companion object {
        const val BACKGROUND = 0xFF15171A.toInt()
    }
}

/** Dims everything outside the ROI box drawn over the camera preview and labels it. */
class GuideOverlay(context: Context) : View(context) {
    private var box = RectF(Roi.FINGER)
    private val px = RectF()
    private var text = ""
    private val shade = Paint().apply { color = 0x88000000.toInt() }
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.WHITE
    }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = sp(14f)
        textAlign = Paint.Align.CENTER
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    fun show(box: RectF, text: String) {
        this.box = RectF(box)
        this.text = text
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val r = px.apply { set(box.left * w, box.top * h, box.right * w, box.bottom * h) }
        canvas.drawRect(0f, 0f, w, r.top, shade)
        canvas.drawRect(0f, r.bottom, w, h, shade)
        canvas.drawRect(0f, r.top, r.left, r.bottom, shade)
        canvas.drawRect(r.right, r.top, w, r.bottom, shade)
        canvas.drawRect(r, stroke)
        canvas.drawText(text, w / 2, (r.top - sp(8f)).coerceAtLeast(sp(16f)), label)
    }
}
