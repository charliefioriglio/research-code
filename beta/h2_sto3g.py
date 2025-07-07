import numpy as np
import matplotlib.pyplot as plt
import time
from ang_grid import repulsion_orientations
from compute_wavefunction import compute_continuum_callable

# ---------------------------------------------
# Define Grid
# ---------------------------------------------
N_ang = 50  # Number of orientations
L = 10
n_pts = 10
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
R_A = np.array([0, 0, (bl_au / 2)])
R_B = np.array([0, 0, -(bl_au / 2)])

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

def rotate_grid(X, Y, Z, alpha, beta, gamma):
    R = rotation_matrix(alpha, beta, gamma)

    # Flatten grid arrays
    shape = X.shape
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)  # shape (3, N)

    # Apply rotation: rotated_coords = R @ coords
    rotated_coords = R @ coords

    # Reshape to original grid shape
    X_rot = rotated_coords[0].reshape(shape)
    Y_rot = rotated_coords[1].reshape(shape)
    Z_rot = rotated_coords[2].reshape(shape)

    return X_rot, Y_rot, Z_rot

# ---------------------------------------------
# Orientational averaging routine
# ---------------------------------------------
def compute_average_amplitudes(X, Y, Z, DO_coeffs_L, DO_coeffs_R, ang_grid, theta_k, phi_k, E_au):
    N = ang_grid.shape[0]
    amps = np.zeros(N, dtype=complex)

    mu = Z

    # Precompute continuum wavefunction on grid
    psi_el_func = compute_continuum_callable(
        m_max=20, n_max=20, E=E_au, a=1.0, D=0.0, L_max=40,
        xi_vals=np.linspace(1.0, 50.0, 500),
        eta_vals=np.linspace(-1.0, 1.0, 200),
        phi_vals=np.linspace(0, 2 * np.pi, 120),
        theta_k=theta_k,
        phi_k=phi_k,
        R_max=40
    )

    psi_el = psi_el_func(X, Y, Z)

    for i in range(N):
        alpha, beta, gamma = ang_grid[i]

        X_rot, Y_rot, Z_rot = rotate_grid(X, Y, Z, alpha, beta, gamma)
        DO_L = build_DO(DO_coeffs_L, X_rot, Y_rot, Z_rot)
        DO_R = build_DO(DO_coeffs_R, X_rot, Y_rot, Z_rot)

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

    ang_grid = repulsion_orientations(n_orientations=N_ang)
    #E_eV = np.linspace(0.01, 3.0, 4)
    E_eV = np.array([2])
    hartree = 27.2114

    results = []

    for E in E_eV:
        E_au = E / hartree

        A_par = compute_average_amplitudes(X, Y, Z, DO_coeffs_sigstar, DO_coeffs_sigstar, ang_grid, 0, 0, E_au)
        A_perp1 = compute_average_amplitudes(X, Y, Z, DO_coeffs_sigstar, DO_coeffs_sigstar, ang_grid, np.pi / 2, 0, E_au)
        A_perp2 = compute_average_amplitudes(X, Y, Z, DO_coeffs_sigstar, DO_coeffs_sigstar, ang_grid, np.pi / 2, np.pi / 2, E_au)

        sigma_par = np.real(np.mean(A_par))
        sigma_perp = 0.5 * np.real(np.mean(A_perp1) + np.mean(A_perp2))

        beta = 2 * (sigma_par - sigma_perp) / (sigma_par + 2 * sigma_perp)

        results.append((E, sigma_par, sigma_perp, beta))

# ---------------------------------------------
# Print Results
# ---------------------------------------------

    print("__________________________________________________________")
    print(f"{'E_KE (eV)':<10} {'beta':>10}")

    for row in results:
        E, sigma_par, sigma_perp, beta = row
        print(f"E = {E:.2f} eV | β = {beta:.4f} | σ_par = {sigma_par:.4e} | σ_perp = {sigma_perp:.4e}")

    print("__________________________________________________________")

    end_time = time.time()
    print(f"\nJob Time: {int(end_time - start_time)} seconds")

# ---------------------------------------------
# Plot Beta vs EKE
# ---------------------------------------------

    results_array = np.array(results)

    plt.figure(figsize=(8,5))
    plt.plot(results_array[:, 0], results_array[:, 3], marker='o')
    plt.xlabel('Photoelectron KE (eV)')
    plt.ylabel(r'$\beta$')
    plt.title('Anisotropy parameter via orientational averaging')
    plt.ylim(-1, 2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
