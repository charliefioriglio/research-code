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

# Hydrogenic orbital ψ_{nlm}
def orb(n, l, m, x, y, z, Z=1):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    rho = 2 * Z * r / n

    norm_radial = np.sqrt((2 * Z / n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    L = genlaguerre(n - l - 1, 2 * l + 1)
    R_nl = norm_radial * np.exp(-rho / 2) * rho**l * L(rho)

    Y_lm = sph_harm(m, l, phi, theta)
    return R_nl * Y_lm

# Dipole matrix element integrands
def integrand_par(k, n, l, m, x, y, z):
    k_hat = k / np.linalg.norm(k)
    psi_el = el(k, x, y, z)
    psi_mol = orb(n, l, m, x, y, z)
    mu_dot_r = k_hat[0]*x + k_hat[1]*y + k_hat[2]*z
    return psi_el * mu_dot_r * psi_mol

def integrand_perp(k, k_el, n, l, m, x, y, z):
    k_hat = k / np.linalg.norm(k)
    psi_el = el(k_el, x, y, z)
    psi_mol = orb(n, l, m, x, y, z)
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
n, l, m = 3, 2, 1
E_eV = np.linspace(0.01, 0.01, 1)  # energy in eV
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
        integrate_3d_simps(integrand_par(kx, n, l, m, X, Y, Z)) +
        integrate_3d_simps(integrand_par(ky, n, l, m, X, Y, Z)) +
        integrate_3d_simps(integrand_par(kz, n, l, m, X, Y, Z))
    )

    # Perpendicular terms
    D_perp += 1/2 * (
        integrate_3d_simps(integrand_perp(kx, ky, n, l, m, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(kx, kz, n, l, m, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(ky, kx, n, l, m, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(ky, kz, n, l, m, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(kz, kx, n, l, m, X, Y, Z)) +
        integrate_3d_simps(integrand_perp(kz, ky, n, l, m, X, Y, Z))
    )

    beta = 2 * (D_par - D_perp) / (D_par + 2 * D_perp)
    beta_vals.append(beta)
print(D_par)
print(D_perp)
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
