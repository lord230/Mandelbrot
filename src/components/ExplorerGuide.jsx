export default function ExplorerGuide() {
    const steps = [
        {
            num: '1',
            title: 'Start at the full view',
            desc: 'The initial view shows Re: [−2.5, 1], Im: [−1.3, 1.3]. The large black "cardioid" body is the main bulb.',
        },
        {
            num: '2',
            title: 'Click to zoom in',
            desc: 'Left-click anywhere to center and zoom in 2×. Try clicking the boundary between black and color for the most detail.',
        },
        {
            num: '3',
            title: 'Right-click to zoom out',
            desc: 'Right-click (or long-press on touch) to zoom out 2× centered on that point. Useful to reorient yourself.',
        },
        {
            num: '4',
            title: 'Raise max iterations',
            desc: 'At deep zooms, increase Max Iterations (try 1500+) to reveal fine filament structures that vanish at low values.',
        },
        {
            num: '5',
            title: 'Switch colormaps',
            desc: 'Different palettes reveal different structures. "Inferno" shows heat-map depth; "Twilight" shows phase symmetry.',
        },
        {
            num: '6',
            title: 'Try these coordinates',
            desc: 'Seahorse Valley: Re ≈ −0.743, Im ≈ 0.127 · Elephant Valley: Re ≈ 0.3, Im ≈ 0 · Mini-brot: Re ≈ −1.77, Im ≈ 0',
        },
        {
            num: '7',
            title: 'Save your discovery',
            desc: 'When you find a beautiful spot, hit "Save PNG" to download a full-resolution image of the current view.',
        },
    ]

    return (
        <div className="info-card glass-card">
            <h3>🧭 Explorer Guide</h3>
            <p>
                The Mandelbrot set boundary has infinite complexity.
                Here's how to navigate and discover its hidden structures.
            </p>

            <h4>Quick Controls</h4>
            <ul>
                <li><strong>Left-click</strong> — Zoom in 2× centered on cursor</li>
                <li><strong>Right-click</strong> — Zoom out 2× centered on cursor</li>
                <li><strong>Reset</strong> — Return to the default full view</li>
                <li><strong>Max Iter slider</strong> — More = finer detail, slower render</li>
                <li><strong>Quality</strong> — Lower = faster preview while exploring</li>
            </ul>

            <h4>Step-by-Step Expedition</h4>
            <ol className="step-list" style={{ listStyle: 'none' }}>
                {steps.map((s) => (
                    <li key={s.num} className="step-item">
                        <div className="step-num">{s.num}</div>
                        <div className="step-body">
                            <strong>{s.title}</strong>
                            <span>{s.desc}</span>
                        </div>
                    </li>
                ))}
            </ol>

            <h4>Notable Locations</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8 }}>
                {[
                    { name: 'Seahorse Valley', re: '−0.7430', im: '0.1270', zoom: 'High' },
                    { name: 'Elephant Valley', re: '0.3000', im: '0.0000', zoom: 'High' },
                    { name: 'Mini-Mandelbrot', re: '−1.7700', im: '0.0000', zoom: 'Very High' },
                    { name: 'Spiral Galaxy', re: '−0.1600', im: '1.0400', zoom: 'Medium' },
                ].map((loc) => (
                    <div
                        key={loc.name}
                        style={{
                            background: 'var(--bg-elevated)',
                            border: '1px solid var(--border-subtle)',
                            borderRadius: 'var(--radius-sm)',
                            padding: '8px 12px',
                        }}
                    >
                        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>
                            {loc.name}
                        </div>
                        <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                            Re ≈ {loc.re} · Im ≈ {loc.im} · Zoom: {loc.zoom}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
