import { useState } from 'react'
import Header from './components/Header'
import MandelbrotCanvas from './components/MandelbrotCanvas'
import ControlPanel from './components/ControlPanel'
import Sidebar from './components/Sidebar'

export default function App() {
  const [palette, setPalette] = useState('twilight')
  const [maxIter, setMaxIter] = useState(800)
  const [quality, setQuality] = useState(1)
  const [resetKey, setResetKey] = useState(0)

  return (
    <div className="app">
      <Header />
      <main className="main-content">
        {/* Left: Canvas + Controls */}
        <div className="viewer-card">
          <MandelbrotCanvas
            palette={palette}
            maxIter={maxIter}
            quality={quality}
            resetKey={resetKey}
          />
          <ControlPanel
            palette={palette}
            onPaletteChange={setPalette}
            maxIter={maxIter}
            onMaxIterChange={setMaxIter}
            quality={quality}
            onQualityChange={setQuality}
            onReset={() => setResetKey(k => k + 1)}
          />
        </div>

        {/* Right: Sidebar */}
        <Sidebar />
      </main>
      <footer className="app-footer">
        Mandelbrot Explorer &mdash; An interactive journey into the complex plane &nbsp;|&nbsp;
        Built with <span style={{ color: 'var(--accent)' }}>React</span> &amp;{' '}
        <span style={{ color: 'var(--purple)' }}>KaTeX</span>
      </footer>
    </div>
  )
}
