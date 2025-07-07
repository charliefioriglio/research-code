import numpy as np
from scipy.special import genlaguerre, factorial, sph_harm
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# Create 3D grid
L = 5
x = np.linspace(-L, L, 100)
y = np.linspace(-L, L, 100)
z = np.linspace(-L, L, 100)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Plane-wave continuum function
def el(k, x, y, z):
    k_dot_r = k[0]*x + k[1]*y + k[2]*z
    return np.exp(1j * k_dot_r)

# O2 Dyson orbital
#Parameters
bl_A = 1.35  # Bond length in angstroms
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

DO_C2 = np.array([-4.44110614e-17, 1.97490563e-16, -7.47775656e-1, -5.13099748e-16, -2.10346844e-16, 1.99423414e-16, -1.83487917e-16, 7.47775656e-1, 1.06160592e-15, 5.17773631e-16])

S1_A = S(C1s, E1, Y, X, Z, R_A) * DO_C[0]
S2_A = S(C2s, E2, Y, X, Z, R_A) * DO_C[1]
Px_A = Px(Cp, E2, Y, X, Z, R_A) * DO_C[2]
Py_A = Py(Cp, E2, Y, X, Z, R_A) * DO_C[3]
Pz_A = Pz(Cp, E2, Y, X, Z, R_A) * DO_C[4]
S1_B = S(C1s, E1, Y, X, Z, R_B) * DO_C[5]
S2_B = S(C2s, E2, Y, X, Z, R_B) * DO_C[6]
Px_B = Px(Cp, E2, Y, X, Z, R_B) * DO_C[7]
Py_B = Py(Cp, E2, Y, X, Z, R_B) * DO_C[8]
Pz_B = Pz(Cp, E2, Y, X, Z, R_B) * DO_C[9]

S1_A2 = S(C1s, E1, Y, X, Z, R_A) * DO_C2[0]
S2_A2 = S(C2s, E2, Y, X, Z, R_A) * DO_C2[1]
Px_A2 = Px(Cp, E2, Y, X, Z, R_A) * DO_C2[2]
Py_A2 = Py(Cp, E2, Y, X, Z, R_A) * DO_C2[3]
Pz_A2 = Pz(Cp, E2, Y, X, Z, R_A) * DO_C2[4]
S1_B2 = S(C1s, E1, Y, X, Z, R_B) * DO_C2[5]
S2_B2 = S(C2s, E2, Y, X, Z, R_B) * DO_C2[6]
Px_B2 = Px(Cp, E2, Y, X, Z, R_B) * DO_C2[7]
Py_B2 = Py(Cp, E2, Y, X, Z, R_B) * DO_C2[8]
Pz_B2 = Pz(Cp, E2, Y, X, Z, R_B) * DO_C2[9]

orb = S1_A + S2_A + Px_A + Py_A + Pz_A + S1_B + S2_B + Px_B + Py_B + Pz_B

orb_flip = S1_A2 + S2_A2 + Px_A2 + Py_A2 + Pz_A2 + S1_B2 + S2_B2 + Px_B2 + Py_B2 + Pz_B2

# Dipole matrix element integrands
def integrand_par(k, x, y, z, mol):
    k_hat = k / np.linalg.norm(k)
    psi_el = el(k, x, y, z)
    psi_mol = mol
    mu_dot_r = k_hat[0]*x + k_hat[1]*y + k_hat[2]*z
    return psi_el * mu_dot_r * psi_mol

def integrand_perp(k, k_el, x, y, z, mol):
    k_hat = k / np.linalg.norm(k)
    psi_el = el(k_el, x, y, z)
    psi_mol = mol
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
        integrate_3d_simps(integrand_par(kx, X, Y, Z, orb)) +
        integrate_3d_simps(integrand_par(ky, X, Y, Z, orb)) +
        integrate_3d_simps(integrand_par(kz, X, Y, Z, orb)) +
        integrate_3d_simps(integrand_par(kx, X, Y, Z, orb_flip)) +
        integrate_3d_simps(integrand_par(ky, X, Y, Z, orb_flip)) +
        integrate_3d_simps(integrand_par(kz, X, Y, Z, orb_flip))
    )

    # Perpendicular terms
    D_perp += 1/2 * (
        integrate_3d_simps(integrand_perp(kx, ky, X, Y, Z, orb)) +
        integrate_3d_simps(integrand_perp(kx, kz, X, Y, Z, orb)) +
        integrate_3d_simps(integrand_perp(ky, kx, X, Y, Z, orb)) +
        integrate_3d_simps(integrand_perp(ky, kz, X, Y, Z, orb)) +
        integrate_3d_simps(integrand_perp(kz, kx, X, Y, Z, orb)) +
        integrate_3d_simps(integrand_perp(kz, ky, X, Y, Z, orb)) +
        integrate_3d_simps(integrand_perp(kx, ky, X, Y, Z, orb_flip)) +
        integrate_3d_simps(integrand_perp(kx, kz, X, Y, Z, orb_flip)) +
        integrate_3d_simps(integrand_perp(ky, kx, X, Y, Z, orb_flip)) +
        integrate_3d_simps(integrand_perp(ky, kz, X, Y, Z, orb_flip)) +
        integrate_3d_simps(integrand_perp(kz, kx, X, Y, Z, orb_flip)) +
        integrate_3d_simps(integrand_perp(kz, ky, X, Y, Z, orb_flip))
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
