export default function HistoryTab() {
    const timeline = [
        {
            year: '1905',
            text: 'Pierre Fatou and Gaston Julia independently study the iteration of complex functions, laying the mathematical foundations.',
        },
        {
            year: '1978',
            text: 'Robert W. Brooks and Peter Matelski first publish an image of the Mandelbrot set as part of a study on Kleinian groups.',
        },
        {
            year: '1980',
            text: 'Benoît Mandelbrot, working at IBM, produces the first high-resolution computer images of the set and brings it to wide scientific attention.',
        },
        {
            year: '1985',
            text: 'A. Douady and J. Hubbard rigorously prove that the Mandelbrot set is connected, a key topological property.',
        },
        {
            year: '1991',
            text: 'Dave Boll discovers the π-in-the-Mandelbrot-set connection: iterations near the neck at c = −0.75 relate to π.',
        },
        {
            year: '1998',
            text: 'Mitsuhiro Shishikura proves the Hausdorff dimension of the boundary is exactly 2, confirming the boundary\'s fractal nature.',
        },
        {
            year: 'Now',
            text: 'GPU-accelerated renders explore depths of 10¹⁵× zoom or more, revealing infinite self-similar structures in real time.',
        },
    ]

    return (
        <div className="info-card glass-card">
            <h3>📜 Historical Timeline</h3>
            <p>
                The Mandelbrot set sits at the intersection of complex analysis, dynamical systems,
                and computer graphics — a century of mathematics made visible.
            </p>
            <div style={{ marginTop: 16 }}>
                {timeline.map((item) => (
                    <div key={item.year} className="history-item">
                        <div className="history-year">{item.year}</div>
                        <div className="history-text">{item.text}</div>
                    </div>
                ))}
            </div>
        </div>
    )
}
