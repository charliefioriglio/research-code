import numpy as np
import matplotlib.pyplot as plt
from ang_grid import repulsion_orientations
import time

# ---------------------------------------------
# Define Grid
# ---------------------------------------------
L = 10
N_ang = 10
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

DO_coeffs_b3g_L = np.array([0, 0, 0, 0, 0, -4.64023371e-01, 0, 0, -3.78623288e-01, 0, 0, -2.82622918e-01, 0, 0, -5.32503344e-03, 0, 0, 0, 0, -5.79841439e-03, 0, 0, 0, 0, 0, 0, 0, 0, 4.64023371e-01, 0, 0, 3.78623288e-01, 0, 0, 2.82622918e-01, 0, 0, -5.32503344e-03, 0, 0, 0, 0, -5.79841439e-03, 0, 0, 0])

DO_coeffs_b3g_R = np.array([0, 0, 0, 0, 0, -4.69555797e-01, 0, 0, -3.80624324e-01, 0, 0, -2.54623001e-01, 0, 0, -5.03930984e-03, 0, 0, 0, 0, -4.80571581e-03, 0, 0, 0, 0, 0, 0, 0, 0, 4.69555797e-01, 0, 0, 3.80624324e-01, 0, 0, 2.54623001e-01, 0, 0, -5.03930984e-03, 0, 0, 0, 0, -4.80571581e-03, 0, 0, 0])

DO_coeffs_b2g_L = np.array([0, 0, 0, 0, -4.75252690e-01, 0, 0, -3.80839945e-01, 0, 0, -2.33638705e-01, 0, 0, 0, 0, 0, -5.61257729e-03, 0, 0, 0, 0, -2.42716567e-03, 0, 0, 0, 0, 0, 4.75252690e-01, 0, 0, 3.80839945e-01, 0, 0, 2.33638705e-01, 0, 0, 0, 0, 0, -5.61257729e-03, 0, 0, 0, 0, -2.42716567e-03, 0])

DO_coeffs_b2g_R = np.array([0, 0, 0, 0, -4.80578093e-01, 0, 0, -3.81713857e-01, 0, 0, -2.08970855e-01, 0, 0, 0, 0, 0, -5.33483553e-03, 0, 0, 0, 0, -1.11917659e-03, 0, 0, 0, 0, 0, 4.80578093e-01, 0, 0, 3.81713857e-01, 0, 0, 2.08970855e-01, 0, 0, 0, 0, 0, -5.33483553e-03, 0, 0, 0, 0, -1.11917659e-03, 0])

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
    
    for i in range(N):
        alpha, beta, gamma = ang_grid[i]

        # Rotate grid into current orientation
        X_rot, Y_rot, Z_rot = rotate_grid(X, Y, Z, alpha, beta, gamma)

        # Evaluate Dyson orbitals on rotated grid (normalized)
        DO_L = build_DO(DO_coeffs_L, X_rot, Y_rot, Z_rot)
        DO_R = build_DO(DO_coeffs_R, X_rot, Y_rot, Z_rot)

        # Use fixed lab-frame polarization and k vector
        mu = pol_lab[0]*X + pol_lab[1]*Y + pol_lab[2]*Z
        psi_el = plane_wave(k_lab, X, Y, Z)

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
        A_par = compute_average_amplitudes(X, Y, Z, DO_coeffs_b3g_L, DO_coeffs_b3g_R, ang_grid, k_par_lab_scaled, pol_lab)
        deg_A_par = compute_average_amplitudes(X, Y, Z, DO_coeffs_b2g_L, DO_coeffs_b2g_R, ang_grid, k_par_lab_scaled, pol_lab)

        A_perp1 = compute_average_amplitudes(X, Y, Z, DO_coeffs_b3g_L, DO_coeffs_b3g_R, ang_grid, k_perp1_lab_scaled, pol_lab)
        A_perp2 = compute_average_amplitudes(X, Y, Z, DO_coeffs_b3g_L, DO_coeffs_b3g_R, ang_grid, k_perp2_lab_scaled, pol_lab)
        deg_A_perp1 = compute_average_amplitudes(X, Y, Z, DO_coeffs_b2g_L, DO_coeffs_b2g_R, ang_grid, k_perp1_lab_scaled, pol_lab)
        deg_A_perp2 = compute_average_amplitudes(X, Y, Z, DO_coeffs_b2g_L, DO_coeffs_b2g_R, ang_grid, k_perp2_lab_scaled, pol_lab)

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
