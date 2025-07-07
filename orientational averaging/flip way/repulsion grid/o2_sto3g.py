import numpy as np
import matplotlib.pyplot as plt
from ang_grid import repulsion_orientations
import time

# ---------------------------------------------
# Define Grid
# ---------------------------------------------
N_ang = 10  # Number of orientations
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

alpha_1S = np.array([1.30709320e+2, 2.38088610e+1, 6.44360830])
coeffs_1S = np.array([1.54328970e-1, 5.35328140e-1, 4.44634540e-1])
alpha_2SP = np.array([5.03315130, 1.16959610, 3.80389000e-1])
coeffs_2S = np.array([-9.99672300e-2, 3.99512830e-1, 7.00115470e-1])
coeffs_2P  = np.array([1.55916270e-1, 6.07683720e-1, 3.9195739e-1])

DO_coeffs_b3g = np.array([0, 0, 0, -7.47775656e-1, 0, 0, 0, 0, 7.47775656e-1, 0])
DO_coeffs_b2g = np.array([0, 0, -7.47775656e-1, 0, 0, 0, 0, 7.47775656e-1, 0, 0])

def build_DO(DO_coeffs, x, y, z):
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

    DO = np.zeros_like(x, dtype=np.float64)
    for alphas, coeffs, a, b, c, center, i in basis_info:
        if DO_coeffs[i] != 0.0:
            ao = AO(alphas, coeffs, a, b, c, center, x, y, z)
            DO += DO_coeffs[i] * ao

    # Normalize final DO
    DO /= np.sqrt(integrate_3d(np.abs(DO)**2, dV))
    return DO

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
def compute_average_amplitudes(X, Y, Z, DO_coeffs_L, DO_coeffs_R, ang_grid, k_lab, pol_lab):
    N = ang_grid.shape[0]
    amps = np.zeros(N, dtype=complex)
    
    mu = pol_lab[0]*X + pol_lab[1]*Y + pol_lab[2]*Z
    psi_el = plane_wave(k_lab, X, Y, Z)

    for i in range(N):
        alpha, beta, gamma = ang_grid[i]

        # Rotate grid into current orientation
        X_rot, Y_rot, Z_rot = rotate_grid(X, Y, Z, alpha, beta, gamma)

        # Evaluate Dyson orbitals on rotated grid (normalized)
        DO_L = build_DO(DO_coeffs_L, X_rot, Y_rot, Z_rot)
        DO_R = build_DO(DO_coeffs_R, X_rot, Y_rot, Z_rot)

        # Compute transition amplitudes A = ∫ ψ_f* μ DO d^3r
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
    pol_lab_z = np.array([0.0, 0.0, 1.0])
    k_z_lab   = np.array([0.0, 0.0, 1.0])
    pol_lab_x = np.array([1.0, 0.0, 0.0])
    pol_lab_y = np.array([0.0, 1.0, 0.0])

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
        k_z_lab_scaled = k_z_lab * k_mag

        # compute amplitudes (sum degenerate orbitals)
        A_par = compute_average_amplitudes(X, Y, Z, DO_coeffs_b3g, DO_coeffs_b3g, ang_grid, k_z_lab_scaled, pol_lab_z)
        deg_A_par = compute_average_amplitudes(X, Y, Z, DO_coeffs_b2g, DO_coeffs_b2g, ang_grid, k_z_lab_scaled, pol_lab_z)

        A_perp1 = compute_average_amplitudes(X, Y, Z, DO_coeffs_b3g, DO_coeffs_b3g, ang_grid, k_z_lab_scaled, pol_lab_x)
        A_perp2 = compute_average_amplitudes(X, Y, Z, DO_coeffs_b3g, DO_coeffs_b3g, ang_grid, k_z_lab_scaled, pol_lab_y)
        deg_A_perp1 = compute_average_amplitudes(X, Y, Z, DO_coeffs_b2g, DO_coeffs_b2g, ang_grid, k_z_lab_scaled, pol_lab_x)
        deg_A_perp2 = compute_average_amplitudes(X, Y, Z, DO_coeffs_b2g, DO_coeffs_b2g, ang_grid, k_z_lab_scaled, pol_lab_y)

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
