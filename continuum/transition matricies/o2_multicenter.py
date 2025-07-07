import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
from scipy.special import spherical_jn, sph_harm, factorial
import scipy.special as sp  # For spherical harmonics
import seaborn as sns
from scipy.special import gamma
from skimage.measure import marching_cubes


m_ph = -1


# Atomic units

# Define Constants
h = 2 * np.pi  # Planck's constant
c = 137  # Speed of light
q = 1 / 27.2  # chanrge of an electron in Hartrees
m_e = 1  # Mass of the electron
wavelength_nm = 500
wavelength_au = wavelength_nm / 52.9e-3  # Wavelength in au
E_rad = (h * c) / wavelength_au  # Energy of ionizing radiation (Hartrees)
E_BE = 0.448 / 27.2  # Electron binding energy in hartrees (v=0 channel)
E_KE = E_rad - E_BE
k = np.sqrt(2 * m_e * E_KE)  # Magnitude of the photoelectron wave vector
bl_A = 1.35  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au

R_A = np.array([0, 0, (bl_au / 2)])
R_B = np.array([0, 0, -(bl_au / 2)])

# Define spatial domain (3D grid)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z = np.linspace(-5, 5, 100)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Norm Procedure
def norm(Psi):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    volume_element = dx * dy * dz
    integral_value = integrate.simps(integrate.simps(integrate.simps(np.abs(Psi)**2, x), y), z)
    return integral_value

# Define Dyson Orbital

# Norm constant for P primitives
def pnorm(e, x, y, z, center):
    x_shift = x - center[0]
    y_shift = y - center[1]
    z_shift = z - center[2]
    r2 = x_shift**2 + y_shift**2 + z_shift**2
    l = 1
    return (2 * e / np.pi)**(3/4) * np.sqrt((4 * e)**l / factorial(2 * l + 1)) * x**l * np.exp(-e * r2)

N_p = 1/np.sqrt(norm(pnorm(10, X, Y, Z, R_A)))

# Define Coefficients and Exponents
E1 = np.array([1.3070932e+2, 2.38088610e+1, 6.4436083])
E2 = np.array([5.03315130, 1.16959610, 3.80389000e-1])
C1s = np.array([1.54328970e-1, 5.35328140e-1, 4.44634540e-1])
C2s = np.array([-9.99672300e-2, 3.99512830e-1, 7.00115470e-1])
Cp = np.array([1.55916270e-1, 6.07683720e-1, 3.9195739e-1])

# Define STO-3G basis functions
def S(c, e, x, y, z, center):
    x_shift = x - center[0]
    y_shift = y - center[1]
    z_shift = z - center[2]
    r2 = x_shift**2 + y_shift**2 + z_shift**2
    return sum(c[i] * (2 * e[i] / np.pi)**(3/4) * np.exp(-e[i] * r2) for i in range(len(c)))

def Px(c, e, x, y, z, center):
    l = 1
    x_shift = x - center[0]
    y_shift = y - center[1]
    z_shift = z - center[2]
    r2 = x_shift**2 + y_shift**2 + z_shift**2
    return N_p * x_shift * sum(c[i] * (2 * e[i] / np.pi)**(3/4) * np.sqrt((4 * e[i])**l / factorial(2 * l + 1)) * np.exp(-e[i] * r2) for i in range(len(c)))

def Py(c, e, x, y, z, center):
    l = 1
    x_shift = x - center[0]
    y_shift = y - center[1]
    z_shift = z - center[2]
    r2 = x_shift**2 + y_shift**2 + z_shift**2
    return N_p * y_shift * sum(c[i] * (2 * e[i] / np.pi)**(3/4) * np.sqrt((4 * e[i])**l / factorial(2 * l + 1)) * np.exp(-e[i] * r2) for i in range(len(c)))

def Pz(c, e, x, y, z, center):
    l = 1
    x_shift = x - center[0]
    y_shift = y - center[1]
    z_shift = z - center[2]
    r2 = x_shift**2 + y_shift**2 + z_shift**2
    return N_p * z_shift * sum(c[i] * (2 * e[i] / np.pi)**(3/4) * np.sqrt((4 * e[i])**l / factorial(2 * l + 1)) * np.exp(-e[i] * r2) for i in range(len(c)))

# Construct Dyson Orbital
DO_C = np.array([-4.44110614e-17, 1.97490563e-16, -5.13099748e-16, -7.47775656e-1, -2.10346844e-16, 1.99423414e-16, -1.83487917e-16, 1.06160592e-15, 7.47775656e-1, 5.17773631e-16])
S1_A = S(C1s, E1, X, Y, Z, R_A) * DO_C[0]
S2_A = S(C2s, E2, X, Y, Z, R_A) * DO_C[1]
Px_A = Px(Cp, E2, X, Y, Z, R_A) * DO_C[2]
Py_A = Py(Cp, E2, X, Y, Z, R_A) * DO_C[3]
Pz_A = Pz(Cp, E2, X, Y, Z, R_A) * DO_C[4]
S1_B = S(C1s, E1, X, Y, Z, R_B) * DO_C[5]
S2_B = S(C2s, E2, X, Y, Z, R_B) * DO_C[6]
Px_B = Px(Cp, E2, X, Y, Z, R_B) * DO_C[7]
Py_B = Py(Cp, E2, X, Y, Z, R_B) * DO_C[8]
Pz_B = Pz(Cp, E2, X, Y, Z, R_B) * DO_C[9]

psi_molecule = S1_A + S2_A + Px_A + Py_A + Pz_A + S1_B + S2_B + Px_B + Py_B + Pz_B

# Check Normalization
DO_norm = norm(psi_molecule)
print(f"Norm: {DO_norm:.6f}")

# Define continuum function

def planewave_component(x, y, z, center, m, l):
    x_shift = x - center[0]
    y_shift = y - center[1]
    z_shift = z - center[2]
    r2 = x_shift**2 + y_shift**2 + z_shift**2
    r = np.sqrt(r2)
    theta = np.arccos(z_shift / r)
    phi = np.arctan2(y_shift, x_shift)
    Y = sph_harm(m, l, phi, theta)
    B = spherical_jn(l, k * r)
    return Y * B

# Transition
# Integrand for symmetric case (el_A + el_B)
def integrand_sym(x, y, z, m_A, m_B, l_A, l_B):
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    el_A = planewave_component(x, y, z, R_A, m_A, l_A)
    el_B = planewave_component(x, y, z, R_B, m_B, l_B)
    psi_el = el_A + el_B
    mu = q * r * sph_harm(m_ph, 1, phi, theta)  # Transition Dipole Moment Opperator

    return np.conj(psi_el) * mu * psi_molecule

# Integrand for antisymmetric case (el_A - el_B)
def integrand_antisym(x, y, z, m_A, m_B, l_A, l_B):
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    el_A = planewave_component(r, theta, phi, R_A, m_A, l_A)
    el_B = planewave_component(r, theta, phi, R_B, m_B, l_B)
    psi_el = el_A - el_B # Antisymmetric combination
    mu = q * r * sph_harm(m_ph, 1, phi, theta)  # Transition Dipole Moment Operator

    return np.conj(psi_el) * mu * psi_molecule

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
            integral_s = integrate.simps(
                integrate.simps(
                    integrate.simps(integrand_sym(X, Y, Z, m_A, m_B, l_A, l_B), x),
                    y
                ), z
            )
            symmetric_matrix[i, j] = np.abs(integral_s) ** 2

            integral_a = integrate.simps(
                integrate.simps(
                    integrate.simps(integrand_antisym(X, Y, Z, m_A, m_B, l_A, l_B), x),
                    y
                ), z
            )
            antisymmetric_matrix[i, j] = np.abs(integral_a) ** 2

    return np.real(symmetric_matrix), np.real(antisymmetric_matrix), lm_pairs

# Define plot routine
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

