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

def integrate_3d(f, dV):
    return np.sum(np.abs(f)**2) * dV

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
# Build AOs
# ---------------------------------------------
def AO_norm(primatives, coeffs, dV):
    norm = 0.0
    n = len(coeffs)
    for i in range(n):
        for j in range(n):
            overlap = np.sum(primatives[i] * primatives[j]) * dV
            norm += coeffs[i] * coeffs[j] * overlap
    return 1.0 / np.sqrt(norm)

# ---------------------------------------------
# Check
# ---------------------------------------------
alphas = np.array([4.94104861e+1, 1.50717731e+1, 5.81515863e+1])
coeffs = np.array([2.19767950e-1, 6.55547360e-1, 2.86573260e-1])

primitives = [gaussian_primitive(alpha, X, Y, Z, 2, 0, 0) for alpha in alphas]

AO = sum(c * p for c, p in zip(coeffs, primitives))
AO_norm_const = AO_norm(primitives, coeffs, dV)
AO_normalized = AO_norm_const * AO
AO_norm_check = integrate_3d(AO_normalized, dV)
print(f"AO norm: {AO_norm_check}")

