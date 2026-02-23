import { useRef, useEffect, useCallback, useState } from 'react'
import MandelbrotWorker from '../mandelbrot.worker.js?worker'

const INITIAL_VIEW = { xmin: -2.5, xmax: 1.0, ymin: -1.3, ymax: 1.3 }

export default function MandelbrotCanvas({ palette, maxIter, quality, resetKey }) {
    const canvasRef = useRef(null)
    const viewRef = useRef({ ...INITIAL_VIEW })
    const workerRef = useRef(null)
    const renderIdRef = useRef(0)       // cancel stale renders
    const downTimeRef = useRef(0)
    const [coords, setCoords] = useState({ ...INITIAL_VIEW })
    const [phase, setPhase] = useState('idle') // idle | coarse | fine | done

    // ── Create worker once ──
    useEffect(() => {
        workerRef.current = new MandelbrotWorker()
        return () => workerRef.current?.terminate()
    }, [])

    // ── Core render function ──
    const render = useCallback(() => {
        const canvas = canvasRef.current
        const worker = workerRef.current
        if (!canvas || !worker) return

        const parent = canvas.parentElement
        if (!parent) return

        const dpr = window.devicePixelRatio || 1
        const rect = parent.getBoundingClientRect()
        const W = Math.floor(rect.width * dpr)
        const H = Math.floor(rect.height * dpr)
        if (W === 0 || H === 0) return

        // Only set the physical pixel buffer size — CSS (width/height: 100%) handles display size.
        // Do NOT set canvas.style.width / canvas.style.height: that mutates the DOM,
        // which triggers ResizeObserver, which causes an infinite grow loop.
        canvas.width = W
        canvas.height = H

        const renderId = ++renderIdRef.current
        const view = { ...viewRef.current }
        setPhase('coarse')

        // ── Worker result handler ──
        worker.onmessage = (e) => {
            // Ignore stale renders
            if (e.data.id !== renderId) return

            const { phase: p, buf, W: bW, H: bH } = e.data

            if (p === 'coarse' || p === 'fine') {
                const ctx = canvas.getContext('2d')
                // Reconstruct ImageData from transferred buffer
                const imageData = new ImageData(new Uint8ClampedArray(buf), bW, bH)
                ctx.putImageData(imageData, 0, 0)
                setPhase(p === 'coarse' ? 'coarse' : 'fine')
                setCoords({ ...viewRef.current })
            }

            if (p === 'done') {
                setPhase('idle')
            }
        }

        // Send work to worker
        worker.postMessage({
            id: renderId,
            W, H,
            xmin: view.xmin, xmax: view.xmax,
            ymin: view.ymin, ymax: view.ymax,
            maxIter,
            palette,
            quality: parseInt(quality) || 1,
        })
    }, [palette, maxIter, quality])

    // ── Render on prop/resetKey changes ──
    useEffect(() => {
        if (resetKey > 0) viewRef.current = { ...INITIAL_VIEW }
        render()
    }, [render, resetKey])

    // ── Resize observer ──
    // Observe the canvas-wrapper (not the canvas itself) to avoid triggering
    // when canvas.width/height changes drive layout shifts.
    const wrapperRef = useRef(null)
    useEffect(() => {
        let rafId
        let lastW = 0, lastH = 0
        const obs = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect
                // Only re-render if size actually changed meaningfully
                if (Math.abs(width - lastW) < 1 && Math.abs(height - lastH) < 1) return
                lastW = width; lastH = height
                cancelAnimationFrame(rafId)
                rafId = requestAnimationFrame(render)
            }
        })
        const wrapper = wrapperRef.current
        if (wrapper) obs.observe(wrapper)
        return () => { obs.disconnect(); cancelAnimationFrame(rafId) }
    }, [render])

    // ── Download handler ──
    useEffect(() => {
        window.__mandelDownload = () => {
            const canvas = canvasRef.current
            if (!canvas) return
            const link = document.createElement('a')
            link.download = `mandelbrot_${Date.now()}.png`
            link.href = canvas.toDataURL('image/png')
            link.click()
        }
    }, [])

    // ── Pointer events ──
    const handlePointerDown = useCallback((ev) => {
        ev.preventDefault()
        downTimeRef.current = performance.now()
    }, [])

    const handlePointerUp = useCallback((ev) => {
        ev.preventDefault()
        const dt = performance.now() - downTimeRef.current
        const canvas = canvasRef.current
        if (!canvas) return
        const rect = canvas.getBoundingClientRect()
        const cx = (ev.clientX - rect.left) / rect.width
        const cy = (ev.clientY - rect.top) / rect.height
        const view = viewRef.current
        const x = view.xmin + cx * (view.xmax - view.xmin)
        const y = view.ymin + cy * (view.ymax - view.ymin)
        const zoomOut = ev.button === 2 || (ev.pointerType !== 'mouse' && dt > 500)
        const factor = zoomOut ? 2.0 : 0.5
        const zx = (view.xmax - view.xmin) * factor
        const zy = (view.ymax - view.ymin) * factor
        viewRef.current = { xmin: x - zx / 2, xmax: x + zx / 2, ymin: y - zy / 2, ymax: y + zy / 2 }
        render()
    }, [render])

    // ── Status indicator style ──
    const statusColor = { idle: 'var(--teal)', coarse: 'var(--gold)', fine: 'var(--accent)' }[phase] || 'var(--text-muted)'
    const statusLabel = { idle: '✓ Ready', coarse: '⚡ Fast preview…', fine: '🎨 Refining…' }[phase] || ''

    return (
        <>
            <div className="canvas-wrapper" ref={wrapperRef}>
                <canvas
                    id="mandelCanvas"
                    ref={canvasRef}
                    className="mandel-canvas"
                    onPointerDown={handlePointerDown}
                    onPointerUp={handlePointerUp}
                    onContextMenu={(e) => e.preventDefault()}
                />
                <div className="canvas-overlay">
                    <div className="zoom-hint">
                        🖱️ Click to zoom in &nbsp;|&nbsp; Right-click to zoom out
                    </div>
                </div>
            </div>

            {/* Coordinate + status bar */}
            <div className="coord-bar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>
                    Re: [<span style={{ color: 'var(--accent)' }}>{coords.xmin.toFixed(6)}</span>, <span style={{ color: 'var(--accent)' }}>{coords.xmax.toFixed(6)}</span>]
                    &nbsp;&nbsp;·&nbsp;&nbsp;
                    Im: [<span style={{ color: 'var(--accent)' }}>{coords.ymin.toFixed(6)}</span>, <span style={{ color: 'var(--accent)' }}>{coords.ymax.toFixed(6)}</span>]
                </span>
                <span style={{ fontSize: 11, color: statusColor, fontWeight: 600, minWidth: 120, textAlign: 'right' }}>
                    {statusLabel}
                </span>
            </div>
        </>
    )
}
