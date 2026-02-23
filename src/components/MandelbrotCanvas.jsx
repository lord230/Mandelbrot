import { useRef, useEffect, useCallback, useState } from 'react'
import MandelbrotWorker from '../mandelbrot.worker.js?worker'

const INITIAL_VIEW = { xmin: -2.5, xmax: 1.0, ymin: -1.3, ymax: 1.3 }
const INITIAL_RANGE = INITIAL_VIEW.xmax - INITIAL_VIEW.xmin // 3.5

// ── URL hash helpers ──────────────────────────────────────────
function viewToHash(v) {
    return `#x0=${v.xmin.toFixed(10)}&x1=${v.xmax.toFixed(10)}&y0=${v.ymin.toFixed(10)}&y1=${v.ymax.toFixed(10)}`
}
function hashToView() {
    try {
        const p = new URLSearchParams(location.hash.slice(1))
        const x0 = parseFloat(p.get('x0'))
        const x1 = parseFloat(p.get('x1'))
        const y0 = parseFloat(p.get('y0'))
        const y1 = parseFloat(p.get('y1'))
        if ([x0, x1, y0, y1].every(Number.isFinite) && x1 > x0 && y1 > y0) {
            return { xmin: x0, xmax: x1, ymin: y0, ymax: y1 }
        }
    } catch { /* ignore */ }
    return null
}

export default function MandelbrotCanvas({ palette, maxIter, quality, resetKey, cycleOffset, onResetRef }) {
    const canvasRef = useRef(null)
    const wrapperRef = useRef(null)
    const viewRef = useRef(hashToView() || { ...INITIAL_VIEW })
    const workerRef = useRef(null)
    const renderIdRef = useRef(0)
    const [coords, setCoords] = useState({ ...viewRef.current })
    const [phase, setPhase] = useState('idle')
    const [isFullscreen, setIsFullscreen] = useState(false)
    const [cursorPos, setCursorPos] = useState(null) // { re, im } or null

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

        canvas.width = W
        canvas.height = H

        const renderId = ++renderIdRef.current
        const view = { ...viewRef.current }

        // Update URL hash
        history.replaceState(null, '', viewToHash(view))
        setCoords({ ...view })
        setPhase('coarse')

        worker.onmessage = (e) => {
            if (e.data.id !== renderId) return
            const { phase: p, buf, W: bW, H: bH } = e.data

            if (p === 'coarse' || p === 'fine') {
                const ctx = canvas.getContext('2d')
                const imageData = new ImageData(new Uint8ClampedArray(buf), bW, bH)
                ctx.putImageData(imageData, 0, 0)
                setPhase(p === 'coarse' ? 'coarse' : 'fine')
                setCoords({ ...viewRef.current })
            }
            if (p === 'done') setPhase('idle')
        }

        worker.postMessage({
            id: renderId,
            W, H,
            xmin: view.xmin, xmax: view.xmax,
            ymin: view.ymin, ymax: view.ymax,
            maxIter,
            palette,
            quality: parseInt(quality) || 1,
            cycleOffset: cycleOffset || 0,
        })
    }, [palette, maxIter, quality, cycleOffset])

    // ── Reset handler exposed upward ──
    useEffect(() => {
        if (onResetRef) onResetRef.current = () => {
            viewRef.current = { ...INITIAL_VIEW }
            render()
        }
    }, [onResetRef, render])

    // ── Render on prop/resetKey changes ──
    useEffect(() => {
        if (resetKey > 0) viewRef.current = { ...INITIAL_VIEW }
        render()
    }, [render, resetKey])

    // ── Resize observer ──
    useEffect(() => {
        let rafId
        let lastW = 0, lastH = 0
        const obs = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect
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

    // ── Fullscreen ──
    useEffect(() => {
        const handler = () => setIsFullscreen(!!document.fullscreenElement)
        document.addEventListener('fullscreenchange', handler)
        return () => document.removeEventListener('fullscreenchange', handler)
    }, [])

    const toggleFullscreen = useCallback(() => {
        const wrapper = wrapperRef.current
        if (!wrapper) return
        if (!document.fullscreenElement) {
            wrapper.requestFullscreen().catch(() => { })
        } else {
            document.exitFullscreen().catch(() => { })
        }
    }, [])

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

    // ── Zoom at canvas center ── (must be defined before keyboard useEffect)
    const zoomAtCenter = useCallback((factor) => {
        const view = viewRef.current
        const cx = (view.xmin + view.xmax) / 2
        const cy = (view.ymin + view.ymax) / 2
        const hw = (view.xmax - view.xmin) * factor / 2
        const hh = (view.ymax - view.ymin) * factor / 2
        viewRef.current = { xmin: cx - hw, xmax: cx + hw, ymin: cy - hh, ymax: cy + hh }
        render()
    }, [render])

    // ── Keyboard shortcuts ──
    useEffect(() => {
        const handleKey = (e) => {
            // Ignore when typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return
            const key = e.key.toLowerCase()
            if (key === 'r') {
                viewRef.current = { ...INITIAL_VIEW }
                render()
            } else if (key === 's') {
                window.__mandelDownload?.()
            } else if (key === 'f') {
                toggleFullscreen()
            } else if (key === '+' || key === '=') {
                e.preventDefault()
                zoomAtCenter(0.5)
            } else if (key === '-') {
                e.preventDefault()
                zoomAtCenter(2.0)
            }
        }
        window.addEventListener('keydown', handleKey)
        return () => window.removeEventListener('keydown', handleKey)
    }, [render, toggleFullscreen, zoomAtCenter])

    // ── Scroll to zoom (toward cursor) ──
    const handleWheel = useCallback((ev) => {
        ev.preventDefault()
        const canvas = canvasRef.current
        if (!canvas) return
        const rect = canvas.getBoundingClientRect()
        const cx = (ev.clientX - rect.left) / rect.width
        const cy = (ev.clientY - rect.top) / rect.height
        const view = viewRef.current
        const mx = view.xmin + cx * (view.xmax - view.xmin)
        const my = view.ymin + cy * (view.ymax - view.ymin)
        const factor = ev.deltaY > 0 ? 1.25 : 0.8
        const newW = (view.xmax - view.xmin) * factor
        const newH = (view.ymax - view.ymin) * factor
        viewRef.current = {
            xmin: mx - cx * newW,
            xmax: mx + (1 - cx) * newW,
            ymin: my - cy * newH,
            ymax: my + (1 - cy) * newH,
        }
        render()
    }, [render])

    // Attach wheel non-passively (passive:false required to call preventDefault)
    useEffect(() => {
        const wrapper = wrapperRef.current
        if (!wrapper) return
        wrapper.addEventListener('wheel', handleWheel, { passive: false })
        return () => wrapper.removeEventListener('wheel', handleWheel)
    }, [handleWheel])

    // ── Pointer drag/click/pan ──
    const dragRef = useRef({ active: false, startX: 0, startY: 0, moved: false })

    const handlePointerDown = useCallback((ev) => {
        ev.preventDefault()
        const canvas = canvasRef.current
        if (!canvas) return
        canvas.setPointerCapture(ev.pointerId)
        dragRef.current = { active: true, startX: ev.clientX, startY: ev.clientY, moved: false, viewSnap: { ...viewRef.current } }
    }, [])

    const handlePointerMove = useCallback((ev) => {
        const canvas = canvasRef.current
        if (!canvas) return
        const rect = canvas.getBoundingClientRect()
        const cx = (ev.clientX - rect.left) / rect.width
        const cy = (ev.clientY - rect.top) / rect.height
        const view = viewRef.current
        setCursorPos({
            re: view.xmin + cx * (view.xmax - view.xmin),
            im: view.ymin + cy * (view.ymax - view.ymin),
        })

        if (!dragRef.current.active) return
        const dx = ev.clientX - dragRef.current.startX
        const dy = ev.clientY - dragRef.current.startY
        if (!dragRef.current.moved && Math.sqrt(dx * dx + dy * dy) > 4) {
            dragRef.current.moved = true
        }
        if (dragRef.current.moved) {
            // Pan: shift view by pixel delta
            const snap = dragRef.current.viewSnap
            const W = rect.width
            const H = rect.height
            const scaleX = (snap.xmax - snap.xmin) / W
            const scaleY = (snap.ymax - snap.ymin) / H
            viewRef.current = {
                xmin: snap.xmin - dx * scaleX,
                xmax: snap.xmax - dx * scaleX,
                ymin: snap.ymin - dy * scaleY,
                ymax: snap.ymax - dy * scaleY,
            }
        }
    }, [])

    const handlePointerUp = useCallback((ev) => {
        ev.preventDefault()
        const drag = dragRef.current
        drag.active = false
        const canvas = canvasRef.current
        if (!canvas) return

        if (drag.moved) {
            // Finalize pan render
            render()
        } else {
            // Click → zoom
            const rect = canvas.getBoundingClientRect()
            const cx = (ev.clientX - rect.left) / rect.width
            const cy = (ev.clientY - rect.top) / rect.height
            const view = viewRef.current
            const x = view.xmin + cx * (view.xmax - view.xmin)
            const y = view.ymin + cy * (view.ymax - view.ymin)
            const zoomOut = ev.button === 2 || ev.altKey
            const factor = zoomOut ? 2.0 : 0.5
            const zx = (view.xmax - view.xmin) * factor
            const zy = (view.ymax - view.ymin) * factor
            viewRef.current = { xmin: x - zx / 2, xmax: x + zx / 2, ymin: y - zy / 2, ymax: y + zy / 2 }
            render()
        }
    }, [render])

    const handlePointerLeave = useCallback(() => setCursorPos(null), [])

    // ── Derived display values ──
    const zoomLevel = (INITIAL_RANGE / (coords.xmax - coords.xmin)).toFixed(2)
    const statusColor = { idle: 'var(--teal)', coarse: 'var(--gold)', fine: 'var(--accent)' }[phase] || 'var(--text-muted)'
    const statusLabel = { idle: '✓ Ready', coarse: '⚡ Preview…', fine: '🎨 Refining…' }[phase] || ''

    return (
        <>
            <div className={`canvas-wrapper${isFullscreen ? ' is-fullscreen' : ''}`} ref={wrapperRef}>
                <canvas
                    id="mandelCanvas"
                    ref={canvasRef}
                    className="mandel-canvas"
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerLeave={handlePointerLeave}
                    onContextMenu={(e) => e.preventDefault()}
                />

                {/* Cursor coordinate HUD */}
                {cursorPos && (
                    <div className="cursor-hud">
                        {cursorPos.re >= 0 ? '+' : ''}{cursorPos.re.toFixed(8)} + {cursorPos.im >= 0 ? '' : ''}{cursorPos.im.toFixed(8)}i
                    </div>
                )}

                {/* Canvas overlay hints */}
                <div className="canvas-overlay">
                    <div className="zoom-hint">
                        🖱 Scroll to zoom &nbsp;·&nbsp; Drag to pan &nbsp;·&nbsp; Click to zoom in &nbsp;·&nbsp; Right-click to zoom out
                    </div>
                </div>

                {/* Fullscreen toggle button */}
                <button
                    className="fullscreen-btn"
                    onClick={toggleFullscreen}
                    title={isFullscreen ? 'Exit fullscreen (F)' : 'Fullscreen (F)'}
                    aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                >
                    {isFullscreen ? '⛶' : '⛶'}
                    <span className="fullscreen-btn-label">{isFullscreen ? 'Exit' : 'Full'}</span>
                </button>
            </div>

            {/* Coordinate + zoom + status bar */}
            <div className="coord-bar">
                <span className="coord-segment">
                    Re: [<span className="coord-val">{coords.xmin.toFixed(6)}</span>, <span className="coord-val">{coords.xmax.toFixed(6)}</span>]
                    &nbsp;·&nbsp;
                    Im: [<span className="coord-val">{coords.ymin.toFixed(6)}</span>, <span className="coord-val">{coords.ymax.toFixed(6)}</span>]
                </span>
                <span className="zoom-badge">×{zoomLevel}</span>
                <span className="status-label" style={{ color: statusColor }}>{statusLabel}</span>
            </div>

            {/* Keyboard shortcut hint */}
            <div className="kbd-hint">
                <kbd>R</kbd> Reset &nbsp;
                <kbd>S</kbd> Save &nbsp;
                <kbd>F</kbd> Fullscreen &nbsp;
                <kbd>+</kbd>/<kbd>-</kbd> Zoom
            </div>
        </>
    )
}
