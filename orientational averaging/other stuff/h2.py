import numpy as np
from scipy.special import genlaguerre, factorial, sph_harm
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# Create 3D grid
L = 50
x = np.linspace(-L, L, 100)
y = np.linspace(-L, L, 100)
z = np.linspace(-L, L, 100)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Plane-wave continuum function
def el(k, x, y, z):
    k_dot_r = k[0]*x + k[1]*y + k[2]*z
    return np.exp(1j * k_dot_r)

# H2 Dyson orbital
#Parameters
bl_A = 1.18  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au
R_A = np.array([0, 0, (bl_au / 2)])
R_B = np.array([0, 0, -(bl_au / 2)])

# Norm Procedure
def norm(f):
    norm = simpson(
        simpson(
            simpson(np.abs(f)**2, x=x, axis=0),  # integrate over x
            x=y, axis=0),                       # integrate over y
        x=z, axis=0)                            # integrate over z
    return norm

# Define Dyson Orbital

# Define Coefficients and Exponents
E1 = np.array([3.42525091, 6.23913730e-1, 1.68855400e-1])
C1 = np.array([1.54328970e-1, 5.35328140e-1, 4.44634540e-1])

# Define STO-3G basis functions
def S(c, e, x, y, z, center):
    x_shift = x - center[0]
    y_shift = y - center[1]
    z_shift = z - center[2]
    r2 = x_shift**2 + y_shift**2 + z_shift**2
    return sum(c[i] * (2 * e[i] / np.pi)**(3/4) * np.exp(-e[i] * r2) for i in range(len(c)))

# Construct Dyson Orbital
#DO_C = np.array([5.98776393e-01, 5.98776393e-01])   # sig coeffs
DO_C = np.array([9.08768850e-01, -9.08768850e-01])   # sig star coeffs
S1_A = S(C1, E1, X, Y, Z, R_A) * DO_C[0]
S1_B = S(C1, E1, X, Y, Z, R_B) * DO_C[1]

orb = S1_A + S1_B

# Check Normalization
DO_norm = norm(orb)
print(f"Norm: {DO_norm:.6f}")


# Dipole matrix element integrands
def integrand_par(k, x, y, z):
    k_hat = k / np.linalg.norm(k)
    psi_el = el(k, x, y, z)
    psi_mol = orb
    mu_dot_r = k_hat[0]*x + k_hat[1]*y + k_hat[2]*z
    return psi_el * mu_dot_r * psi_mol

def integrand_perp(k, k_el, x, y, z):
    k_hat = k / np.linalg.norm(k)
    psi_el = el(k_el, x, y, z)
    psi_mol = orb
    mu_dot_r = k_hat[0]*x + k_hat[1]*y + k_hat[2]*z
    return psi_el * mu_dot_r * psi_mol

# Simpson-based 3D integration
def integrate_3d_simps(f):
    val = simpson(
        simpson(
            simpson(f, x=x, axis=0),  # integrate over x
            x=y, axis=0),                       # integrate over y
        x=z, axis=0)                            # integrate over z
    return np.abs(val)**2

# Parameters
E_eV = np.linspace(0.01, 2, 5)  # energy in eV
hartree = 27.2114
beta_vals = []

for E in E_eV:
    E_au = E / hartree
    k_mag = np.sqrt(2 * E_au)

    kx = k_mag * np.array([1, 0, 0])
    ky = k_mag * np.array([0, 1, 0])
    kz = k_mag * np.array([0, 0, 1])

    D_par = 0
    D_perp = 0
    # Parallel terms
    D_par += (
        integrate_3d_simps(integrand_par(kx, X, Y, Z)) +
        integrate_3d_simps(integrand_par(ky, X, Y, Z)) +
        integrate_3d_simps(integrand_par(kz, X, Y, Z))
    )

    # Perpendicular terms
    D_perp += 1/2 * (
        integrate_3d_simps(integrand_perp(kx, ky, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(kx, kz, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(ky, kx, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(ky, kz, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(kz, kx, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(kz, ky, X, Y, Z))
    )

    beta = 2 * (D_par - D_perp) / (D_par + 2 * D_perp)
    beta_vals.append(beta)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(E_eV, beta_vals, marker='o')
plt.xlabel('Photoelectron Kinetic Energy (eV)')
plt.ylabel(r'$\beta$ parameter')
plt.title(r'$\beta$ vs. Kinetic Energy')
plt.ylim(-1, 2)
plt.grid(True)
plt.tight_layout()
plt.show()
