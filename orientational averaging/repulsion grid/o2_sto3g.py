import numpy as np
import matplotlib.pyplot as plt
from ang_grid import repulsion_orientations
import time

# ---------------------------------------------
# Define Grid
# ---------------------------------------------
N_ang = 10  # Number of orientations
L = 20
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
bl_A = 1.35  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au
R_A = np.array([0, 0, (bl_au / 2)])
R_B = np.array([0, 0, -(bl_au / 2)])

alpha_1S = np.array([1.30709320e+2, 2.38088610e+1, 6.44360830])
coeffs_1S = np.array([1.54328970e-1, 5.35328140e-1, 4.44634540e-1])
alpha_2SP = np.array([5.03315130, 1.16959610, 3.80389000e-1])
coeffs_2S = np.array([-9.99672300e-2, 3.99512830e-1, 7.00115470e-1])
coeffs_2P  = np.array([1.55916270e-1, 6.07683720e-1, 3.9195739e-1])

DO_coeffs_b3g = np.array([0, 0, 0, -7.47775656e-1, 0, 0, 0, 0, 7.47775656e-1, 0])
DO_coeffs_b2g = np.array([0, 0, -7.47775656e-1, 0, 0, 0, 0, 7.47775656e-1, 0, 0])

def build_DO(DO_coeffs):
    basis_info = [
        # Format: (alphas, coeffs, a, b, c, center, DO_coeff index)
        (alpha_1S,  coeffs_1S,  0, 0, 0, R_A, 0),
        (alpha_2SP, coeffs_2S,  0, 0, 0, R_A, 1),
        (alpha_2SP, coeffs_2P,  1, 0, 0, R_A, 2),
        (alpha_2SP, coeffs_2P,  0, 1, 0, R_A, 3),
        (alpha_2SP, coeffs_2P,  0, 0, 1, R_A, 4),
        (alpha_1S,  coeffs_1S,  0, 0, 0, R_B, 5),
        (alpha_2SP, coeffs_2S,  0, 0, 0, R_B, 6),
        (alpha_2SP, coeffs_2P,  1, 0, 0, R_B, 7),
        (alpha_2SP, coeffs_2P,  0, 1, 0, R_B, 8),
        (alpha_2SP, coeffs_2P,  0, 0, 1, R_B, 9),
    ]

    DO = np.zeros_like(X)
    for alphas, coeffs, a, b, c, center, i in basis_info:
        if DO_coeffs[i] != 0.0:  # Skip zero coefficients
            ao = AO(alphas, coeffs, a, b, c, center)
            DO += DO_coeffs[i] * ao

    return DO

DO1_L = build_DO(DO_coeffs_b3g)
DO2_L = build_DO(DO_coeffs_b2g)
DO1_R = build_DO(DO_coeffs_b3g)
DO2_R = build_DO(DO_coeffs_b2g)
DO1_L_norm = integrate_3d(np.abs(DO1_L)**2, dV)
DO2_L_norm = integrate_3d(np.abs(DO2_L)**2, dV)
DO1_R_norm = integrate_3d(np.abs(DO1_R)**2, dV)
DO2_R_norm = integrate_3d(np.abs(DO2_R)**2, dV)
DO1_L_renormalized = DO1_L / np.sqrt(DO1_L_norm)
DO2_L_renormalized = DO2_L / np.sqrt(DO2_L_norm)
DO1_R_renormalized = DO1_R / np.sqrt(DO1_R_norm)
DO2_R_renormalized = DO2_R / np.sqrt(DO2_R_norm)

# ---------------------------------------------
# Plane-wave continuum & rotations
# ---------------------------------------------
def plane_wave(k_vec, x, y, z):
    k_dot_r = k_vec[0]*x + k_vec[1]*y + k_vec[2]*z
    return np.exp(1j * k_dot_r)

# ---------------------------------------------
# Build rotation matrix from Euler angles (ZYZ convention)
# ---------------------------------------------
def rotation_matrix(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz1 = np.array([[ ca, -sa, 0], [ sa,  ca, 0], [  0,   0, 1]])
    Ry  = np.array([[ cb,   0, sb], [  0,   1,  0], [-sb,  0, cb]])
    Rz2 = np.array([[ cg, -sg, 0], [ sg,  cg, 0], [  0,   0, 1]])
    return Rz1 @ Ry @ Rz2

# ---------------------------------------------
# Orientational averaging routine
# ---------------------------------------------
def compute_average_amplitudes(X, Y, Z, DO_L, DO_R, ang_grid, k_lab, pol_lab, mode='parallel'):
    """
    mode = 'parallel' or 'perpendicular'
    For 'parallel': k_lab is along lab z
    For 'perpendicular': k_lab is one of lab x or lab y
    pol_lab remains (0,0,1)
    """
    N = ang_grid.shape[0]
    amps = np.zeros(N, dtype=complex)
    for i in range(N):
        a, b, g = ang_grid[i]
        R = rotation_matrix(a, b, g)
        # rotate polarization and k into molecular frame
        eps_mol = R.T @ pol_lab
        k_mol   = R.T @ k_lab
        # dipole operator mu·eps = eps_x x + eps_y y + eps_z z
        mu = eps_mol[0]*X + eps_mol[1]*Y + eps_mol[2]*Z
        # plane wave in molecular frame
        psi_el = plane_wave(k_mol, X, Y, Z)
        # transition amplitude A = ∫ psi_f * mu * DO d^3r
        integrand_L = np.conj(DO_L) * mu * psi_el
        integrand_R = np.conj(psi_el) * mu * DO_R
        A_L = integrate_3d(integrand_L, dV)
        A_R = integrate_3d(integrand_R, dV)
        amps[i] = A_L * A_R
    return amps

# ---------------------------------------------
# Main script
# ---------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    # lab-frame vectors
    pol_lab = np.array([0.0, 0.0, 1.0])
    k_par_lab   = np.array([0.0, 0.0, 1.0])
    k_perp1_lab = np.array([1.0, 0.0, 0.0])
    k_perp2_lab = np.array([0.0, 1.0, 0.0])

    # Euler angle grid
    ang_grid = repulsion_orientations(n_orientations=N_ang)

    # photoelectron energies (eV)
    E_eV = np.linspace(0.01, 8.0, 10)
    hartree = 27.2114

    results = []

    for E in E_eV:
        E_au = E / hartree
        k_mag = np.sqrt(2 * E_au)

        # scale k_lab directions
        k_par_lab_scaled = k_par_lab * k_mag
        k_perp1_lab_scaled = k_perp1_lab * k_mag
        k_perp2_lab_scaled = k_perp2_lab * k_mag

        # compute amplitudes (sum degenerate orbitals)
        A_par = compute_average_amplitudes(X, Y, Z, DO1_L_renormalized, DO1_R_renormalized, ang_grid, k_par_lab_scaled, pol_lab, mode='parallel')
        deg_A_par = compute_average_amplitudes(X, Y, Z, DO2_L_renormalized, DO2_R_renormalized, ang_grid, k_par_lab_scaled, pol_lab, mode='parallel')

        A_perp1 = compute_average_amplitudes(X, Y, Z, DO1_L_renormalized, DO1_R_renormalized,ang_grid, k_perp1_lab_scaled, pol_lab, mode='perpendicular')
        A_perp2 = compute_average_amplitudes(X, Y, Z, DO1_L_renormalized, DO1_R_renormalized,ang_grid, k_perp2_lab_scaled, pol_lab, mode='perpendicular')
        deg_A_perp1 = compute_average_amplitudes(X, Y, Z, DO2_L_renormalized, DO2_R_renormalized,ang_grid, k_perp1_lab_scaled, pol_lab, mode='perpendicular')
        deg_A_perp2 = compute_average_amplitudes(X, Y, Z, DO2_L_renormalized, DO2_R_renormalized,ang_grid, k_perp2_lab_scaled, pol_lab, mode='perpendicular')

        # intensities
        sigma_par = np.mean(A_par) + np.mean(deg_A_par)
        sigma_perp = 0.5 * (np.mean(A_perp1) + np.mean(A_perp2) + np.mean(deg_A_perp1) + np.mean(deg_A_perp2))
        sig_tot = (4 * np.pi / 3) * (sigma_par + 2 * sigma_perp)

        # anisotropy parameter
        beta = 2 * (sigma_par - sigma_perp) / (sigma_par + 2 * sigma_perp)

        results.append((E, sigma_par, sigma_perp, sig_tot, np.real(beta)))

# ---------------------------------------------
# Print Results
# ---------------------------------------------
    print("_____________________________")
    print(f"{'E_KE (eV)':<10} {'beta':>10}")

    for row in results:
        E, _, _, _, beta = row
        print(f"{E:10.6f} {beta:10.6f}")

    print("_____________________________")

    end_time = time.time()
    print(f"\nJob Time: {int(end_time - start_time)} seconds")

# ---------------------------------------------
# Plot Beta vs EKE
# ---------------------------------------------
    results_array = np.array(results)

    plt.figure(figsize=(8,5))
    plt.plot(results_array[:, 0], results_array[:, 4], marker='o')
    plt.xlabel('Photoelectron KE (eV)')
    plt.ylabel(r'$\beta$')
    plt.title('Anisotropy parameter via orientational averaging')
    plt.ylim(-1, 2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
