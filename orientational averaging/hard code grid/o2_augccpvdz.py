import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# Define Grid
# ---------------------------------------------
L = 20
n_pts = 100
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

DO_coeffs_b3g = np.array([0, 0, 0, 0, 0, -4.64023371e-01, 0, 0, -3.78623288e-01, 0, 0, -2.82622918e-01, 0, 0, -5.32503344e-03, 0, 0, 0, 0, -5.79841439e-03, 0, 0, 0, 0, 0, 0, 0, 0, 4.64023371e-01, 0, 0, 3.78623288e-01, 0, 0, 2.82622918e-01, 0, 0, -5.32503344e-03, 0, 0, 0, 0, -5.79841439e-03, 0, 0, 0])

DO_coeffs_b2g = np.array([0, 0, 0, 0, -4.75252690e-01, 0, 0, -3.80839945e-01, 0, 0, -2.33638705e-01, 0, 0, 0, 0, 0, -5.61257729e-03, 0, 0, 0, 0, -2.42716567e-03, 0, 0, 0, 0, 0, 4.75252690e-01, 0, 0, 3.80839945e-01, 0, 0, 2.33638705e-01, 0, 0, 0, 0, 0, -5.61257729e-03, 0, 0, 0, 0, -2.42716567e-03, 0])

def build_DO(DO_coeffs):
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

    DO = np.zeros_like(X)
    for alphas, coeffs, a, b, c, center, i in basis_info:
        if DO_coeffs[i] != 0.0:
            ao = AO(alphas, coeffs, a, b, c, center)
            DO += DO_coeffs[i] * ao

    # Handle weird d orbitals manually
    # d_z2 = 1/2 * [2zz - xx - yy]
    if DO_coeffs[15] != 0.0:
        ao_zz = AO(alpha_4D_Ag, coeffs_4D_Ag, 0, 0, 2, R_A)
        ao_xx = AO(alpha_4D_Ag, coeffs_4D_Ag, 2, 0, 0, R_A)
        ao_yy = AO(alpha_4D_Ag, coeffs_4D_Ag, 0, 2, 0, R_A)
        DO += DO_coeffs[15] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    # d_x2_y2 = sqrt(3)/2 * (xx - yy)
    if DO_coeffs[17] != 0.0:
        ao_xx = AO(alpha_4D_Ag, coeffs_4D_Ag, 2, 0, 0, R_A)
        ao_yy = AO(alpha_4D_Ag, coeffs_4D_Ag, 0, 2, 0, R_A)
        DO += DO_coeffs[17] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

    if DO_coeffs[20] != 0.0:
        ao_zz = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 0, 2, R_A)
        ao_xx = AO(alpha_6D_Ag, coeffs_6D_Ag, 2, 0, 0, R_A)
        ao_yy = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 2, 0, R_A)
        DO += DO_coeffs[20] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    if DO_coeffs[22] != 0.0:
        ao_xx = AO(alpha_6D_Ag, coeffs_6D_Ag, 2, 0, 0, R_A)
        ao_yy = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 2, 0, R_A)
        DO += DO_coeffs[22] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

    if DO_coeffs[38] != 0.0:
        ao_zz = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 0, 2, R_B)
        ao_xx = AO(alpha_6D_Ag, coeffs_6D_Ag, 2, 0, 0, R_B)
        ao_yy = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 2, 0, R_B)
        DO += DO_coeffs[38] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    if DO_coeffs[40] != 0.0:
        ao_xx = AO(alpha_6D_Ag, coeffs_6D_Ag, 2, 0, 0, R_B)
        ao_yy = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 2, 0, R_B)
        DO += DO_coeffs[40] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

    if DO_coeffs[43] != 0.0:
        ao_zz = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 0, 2, R_B)
        ao_xx = AO(alpha_6D_Ag, coeffs_6D_Ag, 2, 0, 0, R_B)
        ao_yy = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 2, 0, R_B)
        DO += DO_coeffs[43] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    if DO_coeffs[45] != 0.0:
        ao_xx = AO(alpha_6D_Ag, coeffs_6D_Ag, 2, 0, 0, R_B)
        ao_yy = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 2, 0, R_B)
        DO += DO_coeffs[45] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

    return DO

DO1 = build_DO(DO_coeffs_b3g)
DO2 = build_DO(DO_coeffs_b2g)
DO1_norm = integrate_3d(np.abs(DO1)**2, dV)
DO2_norm = integrate_3d(np.abs(DO2)**2, dV)
DO1_renormalized = DO1 / np.sqrt(DO1_norm)
DO2_renormalized = DO2 / np.sqrt(DO2_norm)


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
def compute_average_amplitudes(X, Y, Z, DO, ang_grid, k_lab, pol_lab, mode='parallel'):
    """
    mode = 'parallel' or 'perpendicular'
    For 'parallel': k_lab is along lab z
    For 'perpendicular': k_lab is one of lab x or lab y
    pol_lab remains (0,0,1)
    """
    N = ang_grid.shape[1]
    amps = np.zeros(N, dtype=complex)
    for i in range(N):
        a, b, g = ang_grid[:, i]
        R = rotation_matrix(a, b, 0)
        # rotate polarization and k into molecular frame
        eps_mol = R.T @ pol_lab
        k_mol   = R.T @ k_lab
        # dipole operator mu·eps = eps_x x + eps_y y + eps_z z
        mu = eps_mol[0]*X + eps_mol[1]*Y + eps_mol[2]*Z
        # plane wave in molecular frame
        psi_f = plane_wave(k_mol, X, Y, Z)
        # transition amplitude A = ∫ psi_f * mu * DO d^3r
        integrand = psi_f * mu * DO
        A = integrate_3d(integrand, dV)
        amps[i] = A
    return amps

# ---------------------------------------------
# Main script
# ---------------------------------------------
if __name__ == '__main__':

    # lab-frame vectors
    pol_lab = np.array([0.0, 0.0, 1.0])
    k_par_lab   = np.array([0.0, 0.0, 1.0])
    k_perp1_lab = np.array([1.0, 0.0, 0.0])
    k_perp2_lab = np.array([0.0, 1.0, 0.0])

    # Euler angle grid (3, N)
    ang_grid = np.array([
        [1.07393, 3.31052, 1.74962, 1.64604, 6.10750, 1.75961, 5.53027, 5.52842, 2.28479, 3.58752, 1.66523, 0.84226, 3.29536, 5.57673, 2.47174, 0.92423, 3.93291, 3.31288, 1.90607, 5.84085, 0.38988, 3.04104, 3.73432, 0.06801, 4.53341, 3.95200, 4.40989, 3.00258, 0.72053, 3.05150,  3.43130, 4.27813, 0.33053, 0.21824, 1.26162, 2.18568, 4.36603, 0.72968, 0.09293, 1.39264, 0.61877, 3.83971, 1.33503, 2.72987, 5.64007, 0.64928, 4.61242, 3.37637, 5.84322, 3.88503, 3.30267, 2.80485, 5.96564, 3.63200, 3.21888, 5.15715, 6.16719, 5.35457, 0.54376, 4.22616, 4.85400, 3.31674, 1.37092, 4.28857, 1.13002, 0.71738, 0.54974, 0.55092, 1.24132, 3.64290, 1.64918, 0.21685, 3.08618, 0.92048, 2.12227, 2.48303, 5.11643, 4.98922, 5.85691, 5.20647, 0.55287, 4.78971, 0.20077, 1.62743, 1.16900, 5.35471, 3.02272, 6.06776, 2.18861, 4.82467, 5.11437, 2.76769, 1.87105, 4.61828, 4.49672, 3.57552, 2.38265, 2.76245, 4.72005, 4.04847, 4.67308, 0.03871, 5.47292, 0.39803, 5.04133, 1.91134, 2.47181, 2.18850, 0.02788, 1.64398, 2.99010, 5.39785, 2.22545, 0.13728, 4.95468, 3.95088, 4.48075, 3.89107, 5.28079, 1.04117, 1.52149, 6.18079, 5.73604, 1.91169, 5.92459, 5.51662, 3.78850, 2.46795, 4.85331, 2.43693, 0.84941, 4.11191, 1.05014, 3.60250, 2.66381, 5.14617, 1.60611, 5.85919, 6.20983, 3.03865, 5.79858, 4.28145, 2.75753, 4.16086, 5.81163, 4.79705, 1.95620, 2.56970, 1.91953, 1.30330],
        [2.37487, 2.01802, 0.19298, 1.42732, 0.94148, 2.87768, 0.43346, 2.23837, 2.38175, 1.86742, 0.48430, 1.47576, 2.33447, 1.61340, 1.85509, 2.01268, 0.75689, 1.06096, 1.28682, 2.45608, 1.22613, 0.85503, 2.48750, 2.59037, 1.12758, 1.08387, 1.71823, 2.89856, 1.19880, 1.82581, 0.72273, 2.03217, 2.31736, 1.47451, 2.64602, 1.75559, 0.58855, 1.75632, 1.74660, 1.57420, 0.64023, 1.67455, 1.85446, 1.02968, 1.28404, 2.56314, 1.94129, 0.21312, 0.13739, 1.37820, 1.37309, 2.42800, 1.21872, 2.18767, 2.62084, 2.30207, 2.00292, 1.40952, 1.48686, 1.22480, 2.13311, 1.69066, 1.25267, 2.52738, 1.45655, 2.25519, 2.00568, 2.85707, 2.12000, 0.96483, 1.75524, 2.03425, 0.51224, 0.89061, 0.67305, 1.24734, 0.87917, 0.58117, 1.50352, 2.01029, 0.93519, 1.36289, 0.97133, 0.77734, 0.64195, 2.56591, 1.19311, 1.72994, 1.16124, 2.76802, 1.23525, 1.94278, 2.56288, 0.31374, 2.25460, 1.54648, 0.91810, 1.35613, 1.64956, 2.27038, 0.80996, 0.41045, 0.74431, 1.74535, 1.53611, 1.60435, 1.55181, 1.45794, 1.21354, 1.09534, 2.15902, 1.07567, 2.06928, 0.69204, 1.83188, 2.77746, 1.42929, 1.99852, 1.71301, 1.17178, 2.35523, 1.46674, 0.97673, 2.24095, 0.67027, 1.93475, 0.47169, 2.66075, 1.07081, 0.42167, 0.36992, 1.78975, 1.74413, 1.25456, 0.70579, 3.06199, 2.05766, 2.12962, 2.29422, 1.51664, 1.83453, 0.90821, 1.65087, 1.50476, 2.78176, 2.47069, 0.94743, 2.17897, 1.92705, 0.94949],
        [0.00649606, 0.00677567, 0.00640311, 0.00666473, 0.00669922, 0.00663394, 0.00674343, 0.00667127, 0.00668776, 0.00676721, 0.00665191, 0.00663109, 0.00663458, 0.00674962, 0.00667875, 0.00676506, 0.00663678, 0.00664742, 0.00663493, 0.00677988, 0.00666949, 0.00678447, 0.00678442, 0.00674655, 0.00660888, 0.00662420, 0.00663710, 0.00638865, 0.00674299, 0.00667288, 0.00663664, 0.00663669, 0.00669460, 0.00674184, 0.00666331, 0.00668894, 0.00668811, 0.00678161, 0.00666844, 0.00649848, 0.00672334, 0.00663329, 0.00657182, 0.00678480, 0.00678987, 0.00668689, 0.00668108, 0.00665181, 0.00659894, 0.00675097, 0.00676891, 0.00677254, 0.00678260, 0.00672571, 0.00664667, 0.00640289, 0.00656302, 0.00671165, 0.00670240, 0.00655946, 0.00665147, 0.00671663, 0.00668897, 0.00670018, 0.00646668, 0.00646788, 0.00663172, 0.00672750, 0.00657072, 0.00637692, 0.00672365, 0.00674388, 0.00667000, 0.00666926, 0.00675168, 0.00664912, 0.00663779, 0.00677741, 0.00670043, 0.00660064, 0.00656175, 0.00666901, 0.00638311, 0.00668824, 0.00678113, 0.00665192, 0.00672800, 0.00638353, 0.00639049, 0.00675165, 0.00663724, 0.00639167, 0.00676403, 0.00675114, 0.00666951, 0.00662358, 0.00666779, 0.00663807, 0.00668906, 0.00678518, 0.00666891, 0.00666294, 0.00671233, 0.00670227, 0.00677799, 0.00676458, 0.00677420, 0.00666742, 0.00659765, 0.00672893, 0.00665960, 0.00677759, 0.00676543, 0.00657994, 0.00675244, 0.00666695, 0.00672720, 0.00664683, 0.00674483, 0.00669447, 0.00672329, 0.00659747, 0.00678974, 0.00678624, 0.00674990, 0.00666400, 0.00668276, 0.00666590, 0.00637544, 0.00664386, 0.00667255, 0.00637753, 0.00676400, 0.00663239, 0.00669928, 0.00664242, 0.00676106, 0.00672395, 0.00666835, 0.00677843, 0.00658002, 0.00672617, 0.00666076, 0.00662597, 0.00668821, 0.00664327, 0.00664268, 0.00667836, 0.00678659, 0.00674897]
    ])

    N_ang = ang_grid.shape[1]

    # photoelectron energies (eV)
    E_eV = np.linspace(0.01, 8.0, 10)
    hartree = 27.2114
    beta_vals = []

    for E in E_eV:
        E_au = E / hartree
        k_mag = np.sqrt(2 * E_au)
        # scale k_lab directions
        k_par_lab_scaled = k_par_lab * k_mag
        k_perp1_lab_scaled = k_perp1_lab * k_mag
        k_perp2_lab_scaled = k_perp2_lab * k_mag

        # compute amplitudes (sum degenerate orbitals)
        # compute amplitudes
        A_par = compute_average_amplitudes(X, Y, Z, DO1_renormalized, ang_grid, k_par_lab_scaled, pol_lab, mode='parallel')
        deg_A_par = compute_average_amplitudes(X, Y, Z, DO2_renormalized, ang_grid, k_par_lab_scaled, pol_lab, mode='parallel')

        A_perp1 = compute_average_amplitudes(X, Y, Z, DO1_renormalized, ang_grid, k_perp1_lab_scaled, pol_lab, mode='perpendicular')
        A_perp2 = compute_average_amplitudes(X, Y, Z, DO1_renormalized, ang_grid, k_perp2_lab_scaled, pol_lab, mode='perpendicular')
        deg_A_perp1 = compute_average_amplitudes(X, Y, Z, DO2_renormalized, ang_grid, k_perp1_lab_scaled, pol_lab, mode='perpendicular')
        deg_A_perp2 = compute_average_amplitudes(X, Y, Z, DO2_renormalized, ang_grid, k_perp2_lab_scaled, pol_lab, mode='perpendicular')

        # intensities
        sigma_par = np.mean(np.abs(A_par)**2) + np.mean(np.abs(deg_A_par)**2)
        sigma_perp = 0.5 * (np.mean(np.abs(A_perp1)**2) + np.mean(np.abs(A_perp2)**2) + np.mean(np.abs(deg_A_perp1)**2) + np.mean(np.abs(deg_A_perp2)**2))

        # anisotropy parameter
        beta = 2 * (sigma_par - sigma_perp) / (sigma_par + 2 * sigma_perp)
        beta_vals.append(beta)

    # plot
    print(beta_vals)
    plt.figure(figsize=(8,5))
    plt.plot(E_eV, beta_vals, marker='o')
    plt.xlabel('Photoelectron KE (eV)')
    plt.ylabel(r'$\beta$')
    plt.title('Anisotropy parameter via orientational averaging')
    plt.ylim(-1, 2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
