import numpy as np

# ---------------------------------------------
# Define Grid
# ---------------------------------------------
L = 10
n_pts = 100
x = np.linspace(-L, L, n_pts)
y = np.linspace(-L, L, n_pts)
z = np.linspace(-L, L, n_pts)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]
dV = dx * dy * dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# ---------------------------------------------
# Double factorial
# ---------------------------------------------
def double_factorial(n):
    if n <= 0:
        return 1
    else:
        return n * double_factorial(n - 2)

# ---------------------------------------------
# Normalization factor for Cartesian Gaussians
# ---------------------------------------------
def norm_cartesian_gaussian(alpha, a, b, c):
    l = a + b + c
    prefactor = (2 * alpha / np.pi)**(3/4)
    numerator = (4 * alpha)**l
    denom = double_factorial(2*a - 1) * double_factorial(2*b - 1) * double_factorial(2*c - 1)
    return prefactor * np.sqrt(numerator / denom)

# ---------------------------------------------
# Primitive Gaussian with given (a, b, c)
# ---------------------------------------------
def gaussian_primitive(alpha, x, y, z, a, b, c):
    r2 = x**2 + y**2 + z**2
    norm = norm_cartesian_gaussian(alpha, a, b, c)
    return norm * (x**a) * (y**b) * (z**c) * np.exp(-alpha * r2)

# ---------------------------------------------
# Integration (Riemann sum over cube)
# ---------------------------------------------
def integrate_3d(f, dV):
    return np.sum(np.abs(f)**2) * dV

# ---------------------------------------------
# Build and Normalize
# ---------------------------------------------
alpha = 1.0
S = gaussian_primitive(alpha, X, Y, Z, 0, 0, 0)
P = gaussian_primitive(alpha, X, Y, Z, 1, 0, 0)
D = gaussian_primitive(alpha, X, Y, Z, 1, 1, 0)

S_norm = integrate_3d(S, dV)
P_norm = integrate_3d(P, dV)
D_norm = integrate_3d(D, dV)

print(f"S orbital norm: {S_norm}")
print(f"P orbital norm: {P_norm}")
print(f"D orbital norm: {D_norm}")
