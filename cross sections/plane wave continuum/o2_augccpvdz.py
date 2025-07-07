import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, spherical_jn

# ---------------------------------------------
# Define Grid
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
bl_A = 1.35  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au
R_A = np.array([0, 0, (bl_au / 2)])
R_B = np.array([0, 0, -(bl_au / 2)])

alpha_1S = np.array([1.172e+4, 1.759e+3, 4.008e+2, 1.137e+2, 3.703e+1, 1.327e+1, 5.025, 1.013])
coeffs_1S = np.array([7.1e-4, 5.47e-3, 2.7837e-2, 1.048e-1, 2.83062e-1, 4.48719e-1, 2.70952e-1, 1.5458e-1])
alpha_2S = np.array([1.172e+4, 1.759e+3, 4.008e+2, 1.137e+2, 3.703e+1, 1.327e+1, 5.025, 1.013])
coeffs_2S = np.array([-1.6e-4, -1.263e-3, -6.267e-3, -25716e-2, -7.0924e-2, -1.65411e-1, -1.16955e-1, 5.57368e-1])
alpha_3S = np.array([3.023e-1])
coeffs_3S = np.array([1])
alpha_4S = np.array([7.896e-2])
coeffs_4S = np.array([1])
alpha_5P = np.array([1.77e+1, 3.854, 1.046])
coeffs_5P = np.array([4.3018e-2, 2.28913e-1, 5.08728e-1])
alpha_6P = np.array([2.753e-1])
coeffs_6P = np.array([1])
alpha_7P = np.array([6.856e-2])
coeffs_7P = np.array([1])
alpha_8D = np.array([1.185])
coeffs_8D = np.array([1])
alpha_9D = np.array([3.32e-1])
coeffs_9D = np.array([1])

DO_coeffs_L = np.array([0, 0, 0, 0, 0, -4.64023371e-01, 0, 0, -3.78623288e-01, 0, 0, -2.82622918e-01, 0, 0, -5.32503344e-03, 0, 0, 0, 0, -5.79841439e-03, 0, 0, 0, 0, 0, 0, 0, 0, 4.64023371e-01, 0, 0, 3.78623288e-01, 0, 0, 2.82622918e-01, 0, 0, -5.32503344e-03, 0, 0, 0, 0, -5.79841439e-03, 0, 0, 0])

DO_coeffs_R = np.array([0, 0, 0, 0, 0, -4.69555797e-01, 0, 0, -3.80624324e-01, 0, 0, -2.54623001e-01, 0, 0, -5.03930984e-03, 0, 0, 0, 0, -4.80571581e-03, 0, 0, 0, 0, 0, 0, 0, 0, 4.69555797e-01, 0, 0, 3.80624324e-01, 0, 0, 2.54623001e-01, 0, 0, -5.03930984e-03, 0, 0, 0, 0, -4.80571581e-03, 0, 0, 0])

def build_DO(DO_coeffs, x, y, z):
    basis_info = [
        # Format: (alphas, coeffs, a, b, c, center, DO_coeff index)
        (alpha_1S, coeffs_1S, 0, 0, 0, R_A, 0),
        (alpha_2S, coeffs_2S, 0, 0, 0, R_A, 1),
        (alpha_3S, coeffs_3S, 0, 0, 0, R_A, 2),
        (alpha_4S, coeffs_4S, 0, 0, 0, R_A, 3),
        (alpha_5P, coeffs_5P, 1, 0, 0, R_A, 4),
        (alpha_5P, coeffs_5P, 0, 1, 0, R_A, 5),
        (alpha_5P, coeffs_5P, 0, 0, 1, R_A, 6),
        (alpha_6P, coeffs_6P, 1, 0, 0, R_A, 7),
        (alpha_6P, coeffs_6P, 0, 1, 0, R_A, 8),
        (alpha_6P, coeffs_6P, 0, 0, 1, R_A, 9),
        (alpha_7P, coeffs_7P, 1, 0, 0, R_A, 10),
        (alpha_7P, coeffs_7P, 0, 1, 0, R_A, 11),
        (alpha_7P, coeffs_7P, 0, 0, 1, R_A, 12),
        (alpha_8D, coeffs_8D, 1, 1, 0, R_A, 13),
        (alpha_8D, coeffs_8D, 0, 1, 1, R_A, 14),
        #d z^2 done manually
        (alpha_8D, coeffs_8D, 1, 0, 1, R_A, 16),
        #d x^2 - y^2 done manually
        (alpha_9D, coeffs_9D, 1, 1, 0, R_A, 18),
        (alpha_9D, coeffs_9D, 0, 1, 1, R_A, 19),
        #d z^2 done manually
        (alpha_9D, coeffs_9D, 1, 0, 1, R_A, 21),
        #d x^2 - y^2 done manually
        (alpha_1S, coeffs_1S, 0, 0, 0, R_B, 23),
        (alpha_2S, coeffs_2S, 0, 0, 0, R_B, 24),
        (alpha_3S, coeffs_3S, 0, 0, 0, R_B, 25),
        (alpha_4S, coeffs_4S, 0, 0, 0, R_B, 26),
        (alpha_5P, coeffs_5P, 1, 0, 0, R_B, 27),
        (alpha_5P, coeffs_5P, 0, 1, 0, R_B, 28),
        (alpha_5P, coeffs_5P, 0, 0, 1, R_B, 29),
        (alpha_6P, coeffs_6P, 1, 0, 0, R_B, 30),
        (alpha_6P, coeffs_6P, 0, 1, 0, R_B, 31),
        (alpha_6P, coeffs_6P, 0, 0, 1, R_B, 32),
        (alpha_7P, coeffs_7P, 1, 0, 0, R_B, 33),
        (alpha_7P, coeffs_7P, 0, 1, 0, R_B, 34),
        (alpha_7P, coeffs_7P, 0, 0, 1, R_B, 35),
        (alpha_8D, coeffs_8D, 1, 1, 0, R_B, 36),
        (alpha_8D, coeffs_8D, 0, 1, 1, R_B, 37),
        #d z^2 done manually
        (alpha_8D, coeffs_8D, 1, 0, 1, R_B, 39),
        #d x^2 - y^2 done manually
        (alpha_9D, coeffs_9D, 1, 1, 0, R_B, 41),
        (alpha_9D, coeffs_9D, 0, 1, 1, R_B, 42),
        #d z^2 done manually
        (alpha_9D, coeffs_9D, 1, 0, 1, R_B, 44),
        #d x^2 - y^2 done manually
    ]

    DO = np.zeros_like(x, dtype=np.float64)
    for alphas, coeffs, a, b, c, center, i in basis_info:
        if DO_coeffs[i] != 0.0:
            ao = AO(alphas, coeffs, a, b, c, center, x, y, z)
            DO += DO_coeffs[i] * ao

    # Handle weird d orbitals manually
    # d_z2 = 1/2 * [2zz - xx - yy]
    if DO_coeffs[15] != 0.0:
        ao_zz = AO(alpha_8D, coeffs_8D, 0, 0, 2, R_A)
        ao_xx = AO(alpha_8D, coeffs_8D, 2, 0, 0, R_A)
        ao_yy = AO(alpha_8D, coeffs_8D, 0, 2, 0, R_A)
        DO += DO_coeffs[15] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    # d_x2_y2 = sqrt(3)/2 * (xx - yy)
    if DO_coeffs[17] != 0.0:
        ao_xx = AO(alpha_8D, coeffs_8D, 2, 0, 0, R_A)
        ao_yy = AO(alpha_8D, coeffs_8D, 0, 2, 0, R_A)
        DO += DO_coeffs[17] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

    if DO_coeffs[20] != 0.0:
        ao_zz = AO(alpha_9D, coeffs_9D, 0, 0, 2, R_A)
        ao_xx = AO(alpha_9D, coeffs_9D, 2, 0, 0, R_A)
        ao_yy = AO(alpha_9D, coeffs_9D, 0, 2, 0, R_A)
        DO += DO_coeffs[20] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    if DO_coeffs[22] != 0.0:
        ao_xx = AO(alpha_9D, coeffs_9D, 2, 0, 0, R_A)
        ao_yy = AO(alpha_9D, coeffs_9D, 0, 2, 0, R_A)
        DO += DO_coeffs[22] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

    if DO_coeffs[38] != 0.0:
        ao_zz = AO(alpha_9D, coeffs_8D, 0, 0, 2, R_B)
        ao_xx = AO(alpha_9D, coeffs_8D, 2, 0, 0, R_B)
        ao_yy = AO(alpha_9D, coeffs_8D, 0, 2, 0, R_B)
        DO += DO_coeffs[38] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    if DO_coeffs[40] != 0.0:
        ao_xx = AO(alpha_8D, coeffs_8D, 2, 0, 0, R_B)
        ao_yy = AO(alpha_8D, coeffs_8D, 0, 2, 0, R_B)
        DO += DO_coeffs[40] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

    if DO_coeffs[43] != 0.0:
        ao_zz = AO(alpha_9D, coeffs_9D, 0, 0, 2, R_B)
        ao_xx = AO(alpha_9D, coeffs_9D, 2, 0, 0, R_B)
        ao_yy = AO(alpha_9D, coeffs_9D, 0, 2, 0, R_B)
        DO += DO_coeffs[43] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    if DO_coeffs[45] != 0.0:
        ao_xx = AO(alpha_9D, coeffs_9D, 2, 0, 0, R_B)
        ao_yy = AO(alpha_9D, coeffs_9D, 0, 2, 0, R_B)
        DO += DO_coeffs[45] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

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
def calculate_cross_sections(E_photon_grid, Trans_E, FC_factors, x, y, z, DO_coeffs_R, DO_coeffs_L, L_max):
    DO_R = build_DO(DO_coeffs_R, x, y, z)
    DO_L = build_DO(DO_coeffs_L, x, y, z)
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
                        integrand_R = np.conj(psi) * DO_R * (pol[0]*X + pol[1]*Y + pol[2]*Z)
                        integrand_L = psi * np.conj(DO_L) * (pol[0]*X + pol[1]*Y + pol[2]*Z)
                        A_L = integrate_3d(integrand_L, dV)
                        A_R = integrate_3d(integrand_R, dV)
                        A = A_L * A_R
                        total_A += A

            sigma = prefactor * total_A
            weights.append(sigma)

        weights = np.array(weights)
        if np.sum(weights) > 0:
            rel_cross_sections[j, :] = weights / np.sum(weights)

    return np.real(rel_cross_sections)

# ---------------------------------------------
# Calculate and Plot
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
rel_cross_sections = calculate_cross_sections(E_photon_grid, Trans_E, FC_factors, x, y, z, DO_coeffs_R, DO_coeffs_L, L_max=1)

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

# ---------------------------------------------
# Save as CSV
# ---------------------------------------------
import pandas as pd

column_names = ["Photon Energy (eV)"] + [f"v={i}" for i in range(len(Trans_E))]
df = pd.DataFrame(np.column_stack((E_photon_grid, rel_cross_sections)), columns=column_names)
df.to_csv("cross_sections.csv", index=False)
