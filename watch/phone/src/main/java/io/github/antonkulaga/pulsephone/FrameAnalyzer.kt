package io.github.antonkulaga.pulsephone

import android.graphics.ImageFormat
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.RectF
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** One analysed frame: sensor timestamp, ROI channel means (0..255) and the share of clipped pixels. */
class Frame(
    val tNs: Long,
    val r: Double,
    val g: Double,
    val b: Double,
    val clipR: Double,
    val clipG: Double,
    val clipB: Double,
    val clipAny: Double,
)

/** Where in the buffer the means of the latest frame came from (meta.roi). */
data class FrameGeometry(
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val crop: Rect,
    val roi: Rect,
    val format: String,
    val step: Int,
)

/**
 * Per-frame colour means over a region of interest. Runs on the analysis executor; never keeps an image.
 *
 * The region is given in upright (display) fractions and mapped into buffer coordinates with the frame's
 * rotation, either within the full frame or within the crop rect (the visible preview).
 */
class FrameAnalyzer(private val onFrame: (Frame) -> Unit) : ImageAnalysis.Analyzer {
    @Volatile var roi: RectF = Roi.FINGER
    @Volatile var roiInCrop = false
    @Volatile var geometry: FrameGeometry? = null
        private set
    private var row = ByteArray(0)

    override fun analyze(image: ImageProxy) {
        image.use {
            val rotation = it.imageInfo.rotationDegrees
            val base = if (roiInCrop) it.cropRect else Rect(0, 0, it.width, it.height)
            val box = toBuffer(roi, base, rotation)
            if (!box.intersect(0, 0, it.width, it.height) || box.width() < 2 || box.height() < 2) return
            val step = samplingStep(box)
            val acc = Acc()
            val format = when (it.format) {
                PixelFormat.RGBA_8888 -> { rgba(it, box, step, acc); "RGBA_8888" }
                ImageFormat.YUV_420_888 -> { yuv(it, box, step, acc); "YUV_420_888" }
                else -> return
            }
            if (acc.n == 0) return
            geometry = FrameGeometry(it.width, it.height, rotation, Rect(it.cropRect), box, format, step)
            onFrame(acc.frame(it.imageInfo.timestamp))
        }
    }

    private class Acc {
        var n = 0
        var sr = 0L
        var sg = 0L
        var sb = 0L
        var cr = 0
        var cg = 0
        var cb = 0
        var ca = 0

        fun add(r: Int, g: Int, b: Int) {
            n++
            sr += r; sg += g; sb += b
            val hr = r >= CLIP; val hg = g >= CLIP; val hb = b >= CLIP
            if (hr) cr++
            if (hg) cg++
            if (hb) cb++
            if (hr || hg || hb) ca++
        }

        fun frame(tNs: Long): Frame {
            val d = n.toDouble()
            return Frame(tNs, sr / d, sg / d, sb / d, cr / d, cg / d, cb / d, ca / d)
        }
    }

    /** RGBA_8888 as CameraX delivers it: one plane, 4 bytes per pixel, rows padded to rowStride. */
    private fun rgba(image: ImageProxy, box: Rect, step: Int, acc: Acc) {
        val plane = image.planes[0]
        val buf = plane.buffer.duplicate()
        val ps = plane.pixelStride
        val rs = plane.rowStride
        val len = (box.width() - 1) * ps + 4 // up to the last byte of the last pixel, never past the row
        if (row.size < len) row = ByteArray(len)
        val dx = ps * step
        var y = box.top
        while (y < box.bottom) {
            buf.position(y * rs + box.left * ps)
            buf.get(row, 0, len)
            var i = 0
            while (i + 2 < len) {
                acc.add(row[i].toInt() and 0xFF, row[i + 1].toInt() and 0xFF, row[i + 2].toInt() and 0xFF)
                i += dx
            }
            y += step
        }
    }

    /** Fallback for devices that hand out YUV anyway: per-pixel full-range BT.601 (JFIF), as camera YUV is. */
    private fun yuv(image: ImageProxy, box: Rect, step: Int, acc: Acc) {
        val (yp, up, vp) = image.planes
        val yb = yp.buffer
        val ub = up.buffer
        val vb = vp.buffer
        var y = box.top
        while (y < box.bottom) {
            var x = box.left
            while (x < box.right) {
                val l = yb.get(y * yp.rowStride + x * yp.pixelStride).toInt() and 0xFF
                val u = (ub.get((y / 2) * up.rowStride + (x / 2) * up.pixelStride).toInt() and 0xFF) - 128
                val v = (vb.get((y / 2) * vp.rowStride + (x / 2) * vp.pixelStride).toInt() and 0xFF) - 128
                acc.add(
                    clamp(l + 1.402 * v),
                    clamp(l - 0.344136 * u - 0.714136 * v),
                    clamp(l + 1.772 * u),
                )
                x += step
            }
            y += step
        }
    }

    companion object {
        /** 8-bit level counted as clipped; the server uses the same threshold. */
        const val CLIP = 250

        /** About 80k sampled pixels per frame at most: the mean barely changes, the analyser keeps up at 60 fps. */
        private const val MAX_SAMPLES = 80_000

        private fun clamp(v: Double) = v.roundToInt().coerceIn(0, 255)

        private fun samplingStep(box: Rect): Int =
            max(1, ceil(sqrt(box.width().toDouble() * box.height() / MAX_SAMPLES)).toInt())

        /**
         * Maps upright fractions [f] into buffer pixels of [base]. rotationDegrees is how far the buffer must be
         * turned clockwise to be upright, so e.g. at 90 the upright top edge is the buffer's left edge.
         */
        fun toBuffer(f: RectF, base: Rect, rotationDegrees: Int): Rect {
            fun map(u: Float, v: Float): Pair<Float, Float> = when (rotationDegrees) {
                90 -> v to 1 - u
                180 -> 1 - u to 1 - v
                270 -> 1 - v to u
                else -> u to v
            }
            val (x0, y0) = map(f.left, f.top)
            val (x1, y1) = map(f.right, f.bottom)
            val w = base.width()
            val h = base.height()
            return Rect(
                base.left + (min(x0, x1) * w).roundToInt(),
                base.top + (min(y0, y1) * h).roundToInt(),
                base.left + (max(x0, x1) * w).roundToInt(),
                base.top + (max(y0, y1) * h).roundToInt(),
            )
        }
    }
}
