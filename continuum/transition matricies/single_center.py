import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import sph_harm, spherical_jn
k = 1
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
    return np.sum(f) * dV

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
def gaussian_primitive(alpha, x, y, z, a, b, c, center):
    x0, y0, z0 = center
    xs, ys, zs = x-x0, y-y0, z-z0
    r2 = xs**2 + ys**2 + zs**2
    norm = norm_cartesian_gaussian(alpha, a, b, c)
    return norm * (xs**a) * (ys**b) * (zs**c) * np.exp(-alpha * r2)

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

def AO(alphas, coeffs, a, b, c, center):
    primitives = [gaussian_primitive(alpha, X, Y, Z, a, b, c, center) for alpha in alphas]
    AO = sum(c * p for c, p in zip(coeffs, primitives))
    AO_norm_const = AO_norm(primitives, coeffs, dV)
    AO_normalized = AO_norm_const * AO
    return AO_normalized

# ---------------------------------------------
# Build DO
# ---------------------------------------------
bl_A = 1.18  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au
R_A = np.array([0, 0, (bl_au / 2)])
R_B = np.array([0, 0, -(bl_au / 2)])

alpha_1S = np.array([3.42525091, 6.23913730e-1, 1.68855400e-1])
coeffs_1S = np.array([1.54328970e-1, 5.35328140e-1, 4.44634540e-1])

DO_coeffs_sig = np.array([5.98776393e-01, 5.98776393e-01])
DO_coeffs_sigstar = np.array([9.08768850e-01, -9.08768850e-01])

def build_DO(DO_coeffs):
    basis_info = [
        # Format: (alphas, coeffs, a, b, c, center, DO_coeff index)
        (alpha_1S,  coeffs_1S,  0, 0, 0, R_A, 0),
        (alpha_1S,  coeffs_1S,  0, 0, 0, R_B, 1),
    ]

    DO = np.zeros_like(X)
    for alphas, coeffs, a, b, c, center, i in basis_info:
        if DO_coeffs[i] != 0.0:  # Skip zero coefficients
            ao = AO(alphas, coeffs, a, b, c, center)
            DO += DO_coeffs[i] * ao

    return DO


DO = build_DO(DO_coeffs_sigstar)
DO_norm = integrate_3d(np.abs(DO)**2, dV)
DO_renormalized = DO / np.sqrt(DO_norm)

# ---------------------------------------------
# Define Continuum Basis
# ---------------------------------------------
def planewave_component(x, y, z, m, l):
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    Y_r = sph_harm(m, l, phi, theta)
    B = spherical_jn(l, k * r)
    return Y_r * B

# ---------------------------------------------
# Use transition matrix elements to calculate coefficients
# ---------------------------------------------
def integrand(x, y, z, m, l):
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    psi_el = planewave_component(x, y, z, m, l)
    mu = r * sph_harm(0, 1, phi, theta)  # Transition Dipole Moment Opperator
    integrand = psi_el * mu * DO

    return integrand

def compute_transition_dipole_matrix_single(l_max):
    lm_pairs = []
    values = []

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            val = integrate_3d(integrand(X, Y, Z, m, l), dV)
            values.append(np.abs(val)**2)
            lm_pairs.append((l, m))

    return np.array(values), lm_pairs

# Compute
l_max = 3
values, lm_pairs = compute_transition_dipole_matrix_single(l_max)

# ---------------------------------------------
# Plot as a heatmap for visualization
# ---------------------------------------------
def plot_heatmap(values, lm_pairs, l_max, title):
    heatmap_data = np.full((l_max + 1, 2 * l_max + 1), np.nan)

    for val, (l, m) in zip(values, lm_pairs):
        heatmap_data[l, m + l_max] = val  # Shift m to index properly

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2e", cmap="viridis", cbar=True,
                xticklabels=range(-l_max, l_max + 1), yticklabels=range(l_max + 1))
    plt.title(title)
    plt.xlabel("m")
    plt.ylabel("l")
    plt.tight_layout()
    plt.show()


# Plot heatmaps
plot_heatmap(values, lm_pairs, l_max, "Transition Dipole Matrix Elements (Dyson vs. Continuum $Y_{lm}$)")


