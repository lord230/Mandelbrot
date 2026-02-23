// Colormap gradient definitions for the preview swatches
const SWATCH_GRADIENTS = {
    twilight: 'linear-gradient(to right, #1a0a4a, #5c2b8f, #b45fc9, #f0a0e0, #7ec8e3, #2a5fa0, #1a0a4a)',
    turbo: 'linear-gradient(to right, #30123b, #4662d7, #35aafa, #1be4b4, #73fa52, #d5e62b, #f77c22, #c5280c)',
    plasma: 'linear-gradient(to right, #0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921)',
    inferno: 'linear-gradient(to right, #000004, #2d1160, #7a0c6e, #c84e47, #f57d15, #fcffa4)',
    viridis: 'linear-gradient(to right, #440154, #31688e, #35b779, #fde725)',
    rainbow: 'linear-gradient(to right, #ff0000, #ff7700, #ffff00, #00ff00, #0000ff, #8b00ff)',
    neon: 'linear-gradient(to right, #0040ff, #00b4ff, #00ffea, #00ff88, #80ff00)',
    ocean: 'linear-gradient(to right, #001020, #003080, #0070d0, #00c0f0, #a0f0ff)',
    grayscale: 'linear-gradient(to right, #000000, #ffffff)',
}

export default function ControlPanel({
    palette, onPaletteChange,
    maxIter, onMaxIterChange,
    quality, onQualityChange,
    onReset,
}) {
    const handleDownload = () => {
        if (typeof window.__mandelDownload === 'function') {
            window.__mandelDownload()
        }
    }

    return (
        <div className="glass-card controls-card">
            <p className="controls-title">⚙️ Render Controls</p>

            <div className="control-row">
                {/* Colormap */}
                <div className="control-group">
                    <label className="control-label">Colormap</label>
                    <div className="custom-select">
                        <select
                            value={palette}
                            onChange={e => onPaletteChange(e.target.value)}
                            id="paletteSelect"
                        >
                            <option value="twilight">Twilight</option>
                            <option value="turbo">Turbo</option>
                            <option value="plasma">Plasma</option>
                            <option value="inferno">Inferno</option>
                            <option value="viridis">Viridis</option>
                            <option value="rainbow">Rainbow</option>
                            <option value="neon">Neon</option>
                            <option value="ocean">Ocean</option>
                            <option value="grayscale">Grayscale</option>
                        </select>
                    </div>
                    <div
                        className="colormap-swatch"
                        style={{ background: SWATCH_GRADIENTS[palette] }}
                    />
                </div>

                {/* Quality */}
                <div className="control-group">
                    <label className="control-label">Render Quality</label>
                    <div className="custom-select">
                        <select
                            value={quality}
                            onChange={e => onQualityChange(Number(e.target.value))}
                            id="qualitySelect"
                        >
                            <option value={1}>High (1×)</option>
                            <option value={2}>Medium (2×)</option>
                            <option value={4}>Low (4×) — Fast</option>
                        </select>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                        Higher quality increases detail
                    </div>
                </div>
            </div>

            {/* Max Iterations slider */}
            <div className="slider-group">
                <div className="control-label">
                    <span>Max Iterations</span>
                    <span className="control-label-value">{maxIter}</span>
                </div>
                <div className="range-track">
                    <input
                        id="maxIterSlider"
                        type="range"
                        min={50}
                        max={3000}
                        step={50}
                        value={maxIter}
                        onChange={e => onMaxIterChange(Number(e.target.value))}
                        style={{
                            background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${((maxIter - 50) / (3000 - 50)) * 100}%, var(--border-medium) ${((maxIter - 50) / (3000 - 50)) * 100}%, var(--border-medium) 100%)`
                        }}
                    />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                    <span>50 (fast)</span>
                    <span>More iterations → finer detail</span>
                    <span>3000 (slow)</span>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="btn-row">
                <button id="resetBtn" className="btn" onClick={onReset}>
                    ↺ Reset View
                </button>
                <button id="downloadBtn" className="btn btn-primary" onClick={handleDownload}>
                    ⬇ Save PNG
                </button>
            </div>
        </div>
    )
}
