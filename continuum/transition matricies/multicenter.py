import numpy as np
from scipy.special import genlaguerre, factorial, sph_harm, spherical_jn
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------
# 3D integration helper
# ---------------------------------------------
def integrate_3d(f, x, y, z):
    # integrate over x (axis=0), y (axis=1), z (axis=2)
    I_x = simpson(f, x=x, axis=0)
    I_xy = simpson(I_x, x=y, axis=0)
    I_xyz = simpson(I_xy, x=z, axis=0)
    return I_xyz

# ---------------------------------------------
# Define grid
# ---------------------------------------------
L = 50.0
n_pts = 100
x = np.linspace(-L, L, n_pts)
y = np.linspace(-L, L, n_pts)
z = np.linspace(-L, L, n_pts)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# ---------------------------------------------
# Define basis set
# ---------------------------------------------
E1 = np.array([3.42525091, 6.23913730e-1, 1.68855400e-1])
C1 = np.array([1.54328970e-1, 5.35328140e-1, 4.44634540e-1])

#Parameters
bl_A = 1.18  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au
R_A = np.array([0, 0, (bl_au / 2)])
R_B = np.array([0, 0, -(bl_au / 2)])
k = 1

# s-type Gaussian (STO-3G)
def S(c, e, x, y, z, center):
    x0, y0, z0 = center
    xs, ys, zs = x-x0, y-y0, z-z0
    r2 = xs**2 + ys**2 + zs**2
    val = 0.0
    for ci, ei in zip(c, e):
        norm = (2*ei/np.pi)**(3/4)
        val += ci * norm * np.exp(-ei * r2)
    return val

# Molecular coefficients for the Dyson orbital
#DO_C = np.array([5.98776393e-01, 5.98776393e-01])   # sig coeffs
DO_C = np.array([9.08768850e-01, -9.08768850e-01])   # sig star coeffs

# Build Dyson Orbital from primitives on both centers
def build_DO(x, y, z):
    S1_A = S(C1, E1, x, y, z, R_A) * DO_C[0]
    S1_B = S(C1, E1, x, y, z, R_B) * DO_C[1]
    return S1_A + S1_B

DO = build_DO(X, Y, Z)

# ---------------------------------------------
# Define Continuum Basis
# ---------------------------------------------
def planewave_component(x, y, z, center, m, l):
    x0, y0, z0 = center
    xc, yc, zc = x - x0, y - y0, z - z0
    r2 = xc**2 + yc**2 + zc**2
    r = np.sqrt(r2)
    theta = np.arccos(zc / r)
    phi = np.arctan2(yc, xc)
    Y = sph_harm(m, l, phi, theta)
    B = spherical_jn(l, k * r)
    return Y * B

# ---------------------------------------------
# Use transition matrix elements to calculate coefficients
# ---------------------------------------------
def integrand_sym(x, y, z, m_A, m_B, l_A, l_B):
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    el_A = planewave_component(x, y, z, R_A, m_A, l_A)
    el_B = planewave_component(x, y, z, R_B, m_B, l_B)
    psi_el = el_A + el_B
    mu = r * sph_harm(0, 1, phi, theta)  # Transition Dipole Moment Opperator
    integrand = np.conj(psi_el) * mu * DO

    return integrand

def integrand_antisym(x, y, z, m_A, m_B, l_A, l_B):
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    el_A = planewave_component(x, y, z, R_A, m_A, l_A)
    el_B = planewave_component(x, y, z, R_B, m_B, l_B)
    psi_el = el_A - el_B
    mu = r * sph_harm(0, 1, phi, theta)  # Transition Dipole Moment Opperator
    integrand = np.conj(psi_el) * mu * DO

    return integrand

# Compute transition dipole matrices
def compute_expanded_transition_dipole_matrices(l_max):
    num_states = (l_max + 1) ** 2  # Total number of (l, m) pairs
    symmetric_matrix = np.zeros((num_states, num_states), dtype=np.complex128)
    antisymmetric_matrix = np.zeros((num_states, num_states), dtype=np.complex128)

    # Generate (l, m) index mapping
    lm_pairs = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            lm_pairs.append((l, m))

    # Compute matrix elements
    for i, (l_A, m_A) in enumerate(lm_pairs):
        for j, (l_B, m_B) in enumerate(lm_pairs):
            integral_s = integrate_3d(integrand_sym(X, Y, Z, m_A, m_B, l_A, l_B), x, y, z)
            symmetric_matrix[i, j] = np.abs(integral_s)**2

            integral_a = integrate_3d(integrand_antisym(X, Y, Z, m_A, m_B, l_A, l_B), x, y, z)
            antisymmetric_matrix[i, j] = np.abs(integral_a)**2

    return np.real(symmetric_matrix), np.real(antisymmetric_matrix), lm_pairs

# ---------------------------------------------
# Plot as a heatmap for visualization
# ---------------------------------------------
def plot_heatmap(matrix, title, lm_pairs):
    plt.figure(figsize=(10, 8))

    # Avoid log(0) errors by setting a minimum threshold
    matrix = np.where(matrix > 0, matrix, np.min(matrix[matrix > 0]) * 1e-5)

    # Apply log scale
    log_matrix = np.log10(matrix)

    ax = sns.heatmap(log_matrix, annot=False, cmap="viridis", cbar=True)

    # Generate tick labels based on (l, m) pairs
    tick_labels = [f"({l},{m})" for l, m in lm_pairs]
    
    ax.set_xticks(np.arange(len(lm_pairs)) + 0.5)
    ax.set_yticks(np.arange(len(lm_pairs)) + 0.5)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

    plt.title(title + " (Log Scale)")
    plt.xlabel(r"$(l_B, m_B)$")
    plt.ylabel(r"$(l_A, m_A)$")
    plt.show()


# Plot
l_max = 3
symmetric_matrix, antisymmetric_matrix, lm_pairs = compute_expanded_transition_dipole_matrices(l_max)

plot_heatmap(symmetric_matrix, "Expanded Symmetric Transition Dipole Matrix (el_A + el_B)", lm_pairs)
plot_heatmap(antisymmetric_matrix, "Expanded Antisymmetric Transition Dipole Matrix (el_A - el_B)", lm_pairs)

