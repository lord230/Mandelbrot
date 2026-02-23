// =============================================
// Mandelbrot Web Worker
// All heavy computation runs off the main thread
// =============================================

// Pre-computed HSL→RGB conversion
function hslToRgb(h, s, l) {
    s /= 100; l /= 100;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;
    let r1 = 0, g1 = 0, b1 = 0;
    if (h < 60) { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
    else { r1 = c; g1 = 0; b1 = x; }
    return [
        (r1 + m) * 255 | 0,
        (g1 + m) * 255 | 0,
        (b1 + m) * 255 | 0,
    ];
}

// Vectorized color computation — returns [r,g,b] for t in [0,1]
function paletteRGB(name, t) {
    switch (name) {
        case 'twilight': {
            const h = (360 * (0.7 + 0.6 * Math.sin(6.2832 * t))) % 360;
            const l = 50 - 15 * Math.cos(6.2832 * t);
            return hslToRgb(h, 90, l);
        }
        case 'turbo': {
            const r = (255 * Math.max(0, Math.sin(3.14159 * t))) | 0;
            const g = (200 * (0.5 + 0.5 * Math.cos(6.2832 * t))) | 0;
            const b = (255 * (1 - Math.abs(2 * t - 1))) | 0;
            return [r, g, b];
        }
        case 'plasma': {
            const r = (255 * t ** 0.3) | 0;
            const g = (255 * t ** 0.6) | 0;
            const b = (255 * (1 - t) ** 0.8) | 0;
            return [r, g, b];
        }
        case 'inferno': {
            const r = (255 * t ** 0.2) | 0;
            const g = (200 * t ** 0.6) | 0;
            const b = (120 * t ** 1.2) | 0;
            return [r, g, b];
        }
        case 'viridis': {
            const r = (255 * Math.max(0, Math.min(1, 0.267 + 0.785 * (1 - t)))) | 0;
            const g = (255 * Math.max(0, Math.min(1, 0.004 + 1.143 * Math.sqrt(t)))) | 0;
            const b = (255 * Math.max(0, Math.min(1, 0.329 + 0.5 * Math.sin(3.14159 * t)))) | 0;
            return [r, g, b];
        }
        case 'rainbow':
            return hslToRgb((360 * t) | 0, 90, 55);
        case 'neon':
            return hslToRgb((200 + 200 * t) % 360, 100, 40 + 30 * Math.sin(3.14159 * t));
        case 'ocean': {
            const r = (10 * (1 - t)) | 0;
            const g = (180 * (0.2 + 0.8 * t)) | 0;
            const b = (255 * Math.sqrt(t)) | 0;
            return [r, g, b];
        }
        default: { // grayscale
            const v = (255 * t) | 0;
            return [v, v, v];
        }
    }
}

// ── Optimized Mandelbrot iteration
//    Returns { iter, zx2, zy2 } — final orbit values for smooth coloring
//    Returns iter = -1 if point is known to be inside (cardioid / bulb)
function mandelbrotIter(x0, y0, maxIter) {
    // Cardioid check — skips ~40% of screen pixels
    const y02 = y0 * y0;
    const q = (x0 - 0.25) * (x0 - 0.25) + y02;
    if (q * (q + (x0 - 0.25)) <= 0.25 * y02) return { iter: -1, zx2: 0, zy2: 0 };

    // Period-2 bulb check
    if ((x0 + 1) * (x0 + 1) + y02 <= 0.0625) return { iter: -1, zx2: 0, zy2: 0 };

    // Main iteration — track zx2/zy2 so we get |z| for free
    let zx = 0, zy = 0, zx2 = 0, zy2 = 0, iter = 0;
    while (zx2 + zy2 <= 4 && iter < maxIter) {
        zy = 2 * zx * zy + y0;
        zx = zx2 - zy2 + x0;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iter++;
    }
    return { iter, zx2, zy2 };
}

// ── Render a full frame
function renderFrame({ W, H, xmin, xmax, ymin, ymax, maxIter, palette, pxStep, cycleOffset }) {
    const buf = new Uint8ClampedArray(W * H * 4);
    const rx = (xmax - xmin) / W;
    const ry = (ymax - ymin) / H;
    const LOG2 = Math.log(2);

    for (let py = 0; py < H; py += pxStep) {
        const y0 = ymin + py * ry;
        for (let px = 0; px < W; px += pxStep) {
            const x0 = xmin + px * rx;
            const { iter, zx2, zy2 } = mandelbrotIter(x0, y0, maxIter);

            let r = 0, g = 0, b = 0;
            if (iter !== -1 && iter < maxIter) {
                // Munafo smooth coloring — no re-iteration needed, we already have zx2/zy2
                const modZ = Math.sqrt(zx2 + zy2);
                const nu = iter + 1 - Math.log(Math.log(Math.max(modZ, 1e-9))) / LOG2;
                // Apply cycleOffset so the palette can be rotated by the user
                const t = ((Math.max(0, Math.min(1, nu / maxIter)) + (cycleOffset || 0)) % 1 + 1) % 1;
                [r, g, b] = paletteRGB(palette, t);
            }

            // Fill pxStep×pxStep block
            for (let dy = 0; dy < pxStep && py + dy < H; dy++) {
                const row = (py + dy) * W;
                for (let dx = 0; dx < pxStep && px + dx < W; dx++) {
                    const ix = (row + px + dx) * 4;
                    buf[ix] = r; buf[ix + 1] = g; buf[ix + 2] = b; buf[ix + 3] = 255;
                }
            }
        }
    }
    return buf;
}

// ── Worker message handler
self.onmessage = function (e) {
    const { id, W, H, xmin, xmax, ymin, ymax, maxIter, palette, quality, cycleOffset } = e.data;

    // Phase 1: Coarse render (always fast — 4× or quality pixels)
    const coarseStep = Math.max(quality, 4);
    const coarseBuf = renderFrame({ W, H, xmin, xmax, ymin, ymax, maxIter, palette, pxStep: coarseStep, cycleOffset });
    self.postMessage({ id, phase: 'coarse', buf: coarseBuf, W, H }, [coarseBuf.buffer]);

    // Phase 2: If quality < coarseStep, render refined
    if (quality < coarseStep) {
        const fineBuf = renderFrame({ W, H, xmin, xmax, ymin, ymax, maxIter, palette, pxStep: quality, cycleOffset });
        self.postMessage({ id, phase: 'fine', buf: fineBuf, W, H }, [fineBuf.buffer]);
    }

    self.postMessage({ id, phase: 'done' });
};
