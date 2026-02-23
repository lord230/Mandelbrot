import { useEffect, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

function KaTeX({ formula, displayMode = false }) {
    const ref = useRef(null)
    useEffect(() => {
        if (ref.current) {
            katex.render(formula, ref.current, {
                displayMode,
                throwOnError: false,
                errorColor: '#ff6b9d',
            })
        }
    }, [formula, displayMode])
    return <span ref={ref} />
}

function MathBlock({ formula, label }) {
    return (
        <div className="katex-formula">
            <KaTeX formula={formula} displayMode={true} />
            {label && <div className="formula-label">{label}</div>}
        </div>
    )
}

export default function MathExplainer() {
    return (
        <div className="info-card glass-card">
            <h3>📐 The Mathematics</h3>

            {/* Section 1: Core Iteration */}
            <div className="math-section">
                <h4>1 · The Iteration Rule</h4>
                <p>
                    The Mandelbrot set is defined by one deceptively simple recurrence. For each point
                    <KaTeX formula=" c " /> in the complex plane, we repeatedly apply:
                </p>
                <MathBlock
                    formula="z_{n+1} = z_n^2 + c \qquad z_0 = 0"
                    label="The core Mandelbrot iteration"
                />
                <p>
                    A point <KaTeX formula=" c " /> <strong style={{ color: 'var(--text-primary)' }}>belongs to the set</strong> if
                    <KaTeX formula=" |z_n| " /> remains bounded forever. Otherwise, the orbit <em>escapes to infinity</em>.
                </p>
            </div>

            {/* Section 2: Complex Numbers */}
            <div className="math-section">
                <h4>2 · Complex Numbers</h4>
                <p>
                    Each point is a complex number <KaTeX formula=" c = x + iy " />, where <KaTeX formula=" x " /> is the
                    real part (horizontal axis) and <KaTeX formula=" y " /> is the imaginary part (vertical axis).
                    Squaring a complex number follows:
                </p>
                <MathBlock
                    formula="z^2 = (x + iy)^2 = x^2 - y^2 + 2xyi"
                    label="Complex multiplication expands the boundary"
                />
                <p>
                    So the iteration splits into two real recurrences:
                </p>
                <MathBlock
                    formula="x_{n+1} = x_n^2 - y_n^2 + \text{Re}(c) \quad y_{n+1} = 2x_n y_n + \text{Im}(c)"
                    label="Equivalent real-valued computation (what the canvas uses)"
                />
            </div>

            {/* Section 3: Escape Criterion */}
            <div className="math-section">
                <h4>3 · Escape Criterion</h4>
                <p>
                    It can be proven: if at any iteration <KaTeX formula=" |z_n| > 2 " />, the orbit <em>will always escape</em>.
                    This gives us the efficient test:
                </p>
                <MathBlock
                    formula="|z_n|^2 = x_n^2 + y_n^2 > 4 \implies c \notin \mathcal{M}"
                    label="No square root needed — saves computation per pixel"
                />
                <p>
                    Points that <em>never</em> escape up to <KaTeX formula="\text{maxIter}" /> steps are rendered <strong style={{ color: 'var(--text-primary)' }}>black</strong>.
                </p>
            </div>

            {/* Section 4: Smooth Coloring */}
            <div className="math-section">
                <h4>4 · Smooth Coloring</h4>
                <p>
                    Naive coloring creates harsh bands. The <strong style={{ color: 'var(--text-primary)' }}>histogram / Munafo smooth coloring</strong> formula eliminates them:
                </p>
                <MathBlock
                    formula="\nu = n + 1 - \frac{\log(\log |z_n|)}{\log 2}"
                    label="Normalized iteration count — continuous across band boundaries"
                />
                <p>
                    <KaTeX formula="\nu " /> is then mapped to <KaTeX formula="[0,1]" /> and fed into the chosen colormap.
                    This creates the smooth gradient you see at fractal boundaries.
                </p>
            </div>

            {/* Section 5: Fractal Dimension */}
            <div className="math-section">
                <h4>5 · Fractal Dimension</h4>
                <p>
                    The Mandelbrot boundary is a <em>fractal</em> — it has infinite perimeter but zero area.
                    Its <strong style={{ color: 'var(--text-primary)' }}>Hausdorff dimension</strong> is exactly:
                </p>
                <MathBlock
                    formula="\dim_H(\partial \mathcal{M}) = 2"
                    label="The boundary is as 'space-filling' as a 2D region — proven by Shishikura (1998)"
                />
                <p>
                    No matter how far you zoom, new intricate structures — mini-Mandelbrots, spirals,
                    filaments — keep appearing at every scale.
                </p>
            </div>

            {/* Fun fact stats */}
            <div className="stat-grid">
                <div className="stat-pill">
                    <span className="stat-pill-value">∞</span>
                    <span className="stat-pill-label">Zoom depth</span>
                </div>
                <div className="stat-pill">
                    <span className="stat-pill-value">0</span>
                    <span className="stat-pill-label">Area of boundary</span>
                </div>
                <div className="stat-pill">
                    <span className="stat-pill-value">2</span>
                    <span className="stat-pill-label">Hausdorff dim.</span>
                </div>
                <div className="stat-pill">
                    <span className="stat-pill-value">1978</span>
                    <span className="stat-pill-label">First visualized</span>
                </div>
            </div>
        </div>
    )
}
