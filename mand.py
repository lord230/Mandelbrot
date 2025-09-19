import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter=500, smooth=True):
    """
    Compute the Mandelbrot set with escape-time or smooth coloring.
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=complex)
    escape_iter = np.zeros(C.shape, dtype=int)
    absZ = np.zeros(C.shape, dtype=float)

    for k in range(max_iter):
        mask = (escape_iter == 0)
        Z[mask] = Z[mask]**2 + C[mask]
        escaped = mask & (np.abs(Z) > 2)
        escape_iter[escaped] = k
        absZ[escaped] = np.abs(Z[escaped])

    escape_iter[escape_iter == 0] = max_iter

    if smooth:
        absZ[absZ == 0] = np.abs(Z[absZ == 0])
        nu = escape_iter + 1 - np.log(np.log(absZ)) / np.log(2)
        nu[np.isnan(nu) | np.isinf(nu)] = max_iter
        M = nu
    else:
        M = escape_iter

    M[escape_iter == max_iter] = 0
    return M


xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width, height = 1200, 900
max_iter = 1000


M = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter, smooth=True)


plt.figure(figsize=(12,8))
plt.imshow(M, extent=(xmin, xmax, ymin, ymax), origin='lower',
           cmap=cm.twilight_shifted, interpolation="bicubic")
plt.colorbar(label="Escape iteration")

plt.title("Mandelbrot Set – The Beauty of Mathematics\n"
          "Domain (Real axis): [-2, 1], Range (Imag axis): [-1.5, 1.5]\n" 
          "Equation: C = X + 1j * Y",
          fontsize=14, weight="bold")

plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.show()
