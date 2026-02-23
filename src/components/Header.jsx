export default function Header() {
    return (
        <header className="app-header">
            <div className="header-brand">
                <div className="header-logo">🌀</div>
                <div>
                    <h1 className="header-title">Mandelbrot Explorer</h1>
                    <p className="header-subtitle">Interactive journey into the infinite fractal boundary</p>
                </div>
            </div>
            <div className="header-badge">
                <span className="badge badge-math">Mathematics</span>
                <span className="badge badge-interactive">Interactive</span>
            </div>
        </header>
    )
}
