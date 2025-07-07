import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, spherical_jn

# ---------------------------------------------
# Define Grid
# ---------------------------------------------
L = 10
n_pts = 50
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

# Constants
hartree = 27.2114  # eV
c = 137.036        # a.u. speed of light

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
def AO_norm(primitives, coeffs):
    norm = 0.0
    n = len(coeffs)
    for i in range(n):
        for j in range(n):
            overlap = np.sum(primitives[i] * primitives[j]) * dV
            norm += coeffs[i] * coeffs[j] * overlap
    return 1.0 / np.sqrt(norm)

def AO(alphas, coeffs, a, b, c, center, x, y, z):
    primitives = [gaussian_primitive(alpha, x, y, z, a, b, c, center) for alpha in alphas]
    ao = sum(c * p for c, p in zip(coeffs, primitives))
    ao_norm_const = AO_norm(primitives, coeffs)
    return ao_norm_const * ao

# ---------------------------------------------
# Build DO
# ---------------------------------------------
bl_A = 1.18  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au
a = bl_au / 2
R_A = np.array([0, 0, a])
R_B = np.array([0, 0, -a])

alpha_1S = np.array([3.42525091, 6.23913730e-1, 1.68855400e-1])
coeffs_1S = np.array([1.54328970e-1, 5.35328140e-1, 4.44634540e-1])

DO_coeffs_sig = np.array([5.98776393e-01, 5.98776393e-01])
DO_coeffs_sigstar = np.array([9.08768850e-01, -9.08768850e-01])

def build_DO(DO_coeffs, x, y, z):
    basis_info = [
        (alpha_1S, coeffs_1S, 0, 0, 0, R_A, 0),
        (alpha_1S, coeffs_1S, 0, 0, 0, R_B, 1),
    ]

    DO = np.zeros_like(x, dtype=np.float64)
    for alphas, coeffs, a, b, c, center, i in basis_info:
        if DO_coeffs[i] != 0.0:
            ao = AO(alphas, coeffs, a, b, c, center, x, y, z)
            DO += DO_coeffs[i] * ao

    # Normalize final DO
    DO /= np.sqrt(integrate_3d(np.abs(DO)**2, dV))
    return DO

# ---------------------------------------------
# Define Planewave Expansion
# ---------------------------------------------
def r_theta_phi(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi

def planewave_expansion(k, l, m, x, y, z):
    r, theta, phi = r_theta_phi(x, y, z)
    R = spherical_jn(l, k * r)
    Y = sph_harm(m, l, phi, theta)
    return R * Y

# ---------------------------------------------
# Cross section calculation
# ---------------------------------------------
def calculate_cross_sections(E_photon_grid, Trans_E, FC_factors, x, y, z, DO_coeffs, L_max):
    DO = build_DO(DO_coeffs, x, y, z)
    polarizations = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    n_vib = len(Trans_E)
    rel_cross_sections = np.zeros((len(E_photon_grid), n_vib))

    for j, E_photon in enumerate(E_photon_grid):
        weights = []
        for i in range(n_vib):
            E_bind = Trans_E[i]
            eKE = E_photon - E_bind
            if eKE <= 0:
                weights.append(0.0)
                continue
            k_mag = np.sqrt(2 * eKE / hartree)
            fc2 = FC_factors[i]**2
            prefactor = fc2 * (8 * np.pi * k_mag * E_photon / c)

            total_A = 0
            for pol in polarizations:
                for l in range(L_max + 1):
                    for m in range(-l, l + 1):
                        psi = planewave_expansion(k_mag, l, m, x, y, z)
                        integrand = np.conj(psi) * DO * (pol[0]*X + pol[1]*Y + pol[2]*Z)
                        A = np.abs(integrate_3d(integrand, dV))**2
                        total_A += A

            sigma = prefactor * total_A
            weights.append(sigma)

        weights = np.array(weights)
        if np.sum(weights) > 0:
            rel_cross_sections[j, :] = weights / np.sum(weights)

    return rel_cross_sections

# ---------------------------------------------
# Vibrational Data and Plotting
# ---------------------------------------------
vib_transitions = np.array([
    [1.0000, 1.285160e-01],
    [1.2581, 2.390435e-01],
    [1.5161, 3.284288e-01],
    [1.7742, 3.828281e-01],
    [2.0322, 3.999465e-01],
])
Trans_E = vib_transitions[:, 0]
FC_factors = vib_transitions[:, 1]

E_photon_grid = np.linspace(1, 5.0, 50)
rel_cross_sections = calculate_cross_sections(E_photon_grid, Trans_E, FC_factors, x, y, z, DO_coeffs_sig, L_max=1)

# Plotting
plt.figure(figsize=(8, 5))
for i in range(len(Trans_E)):
    plt.plot(E_photon_grid, rel_cross_sections[:, i], label=f'v={i}')
plt.xlabel("Photon Energy (eV)")
plt.ylabel("Relative Cross Section")
plt.legend()
plt.title("Vibrationally Resolved Relative Cross Sections")
plt.grid(True)
plt.tight_layout()
plt.show()