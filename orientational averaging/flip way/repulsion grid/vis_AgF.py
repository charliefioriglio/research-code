import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from math import factorial
from skimage.measure import marching_cubes

# Parameters
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
R_A = np.array([0, 0, 0.872093 / 52.9e-2])
R_B = np.array([0, 0, -1.077907 / 52.9e-2])

alpha_1S_Ag = np.array([4.74452163e+3, 8.64220538e+2, 2.33891805e+2])
coeffs_1S_Ag = np.array([1.54328970e-1, 5.35328140-1, 4.44634540e-1])
alpha_2SP_Ag = np.array([4.14965207e+2, 9.64289900e+1, 3.13617003e+1])
coeffs_2S_Ag = np.array([-9.99672300e-2, 3.99512830e-1, 7.001154702-1])
coeffs_2P_Ag = np.array([1.55916280e-1, 6.07683720e-1, 3.91957390e-1])
alpha_3SP_Ag = np.array([4.94104861e+1, 1.50717731e+1, 5.81515863])
coeffs_3S_Ag = np.array([-2.27763500e-1, 2.17543600e-1, 9.16676960e-1])
coeffs_3P_Ag = np.array([4.95151000e-3, 5.77766470e-1, 4.84646040e-1])
alpha_4D_Ag = np.array([4.94104861e+1, 1.50717731e+1, 5.81515863e+1])
coeffs_4D_Ag = np.array([2.19767950e-1, 6.55547360e-1, 2.86573260e-1])
alpha_5SP_Ag = np.array([5.29023045, 2.05998832,  9.06811930e-1])
coeffs_5S_Ag = np.array([-3.30610060e-1, 5.76109500e-2, 1.15578745])
coeffs_5P_Ag = np.array([-1.28392760e-1, 5.85204760e-1, 5.43944200e-1])
alpha_6D_Ag = np.array([3.28339567, 1.27853725, 5.62815250e-1])
coeffs_6D_Ag = np.array([1.25066210e-1, 6.68678560e-1, 3.05246820e-1])
alpha_7SP_Ag = np.array([4.37080480e-1, 2.35340820e-1, 1.03954180e-1])
coeffs_7S_Ag = np.array([-3.8426426e-1, -1.97256740e-1, 1.37549551])
coeffs_7P_Ag = np.array([-3.48169150e-1, 6.29032370e-1, 6.66283270e-1])
alpha_1S_F = np.array([1.66679130e+2, 3.03608120e+1, 8.21682070])
coeffs_1S_F = np.array([1.54328970e-1, 5.35328140e-1, 4.44634540e-1])
alpha_2SP_F = np.array([6.46480320, 1.50228120, 4.88588500e-1])
coeffs_2S_F = np.array([-9.99672300e-2, 3.99512830e-1, 7.00115470e-1])
coeffs_2P_F = np.array([1.55916270e-1, 6.07683720e-1, 3.919573902-1])

DO_coeffs_AgF_L = np.array([9.14529141e-03, -2.69537956e-02, 0, 0, 2.89929386e-02, 5.85023715e-02, 0, 0, -8.04092502e-02, 0, 0, -1.50307315e-01, 0, 0, -1.17098135e-01, 0, 0, 1.90945673e-01, 0, 0, 5.51622500e-01, 0, 0, 5.06467116e-01, 0, 0, -5.96318801e-01, -5.97134365e-03, 7.04717812e-02, 0, 0, 3.39488329e-01])

DO_coeffs_AgF_R = np.array([9.15835831e-03, -2.69843543e-02, 0, 0, 3.02739714e-02, 5.83796178e-02, 0, 0, -7.69926002e-02, 0, 0, -1.52940363e-01, 0, 0, -1.17165422e-01, 0, 0, 1.82592632e-01, 0, 0, 5.61511613e-01, 0, 0, 5.11896524e-01, 0, 0, -5.57732243e-01, -4.77359388e-03, 5.90556233e-02, 0, 0, 3.80592224e-01])

def build_DO(DO_coeffs, x, y, z, R_Ag, R_F):
    basis_info = [
        # Format: (alphas, coeffs, a, b, c, center, DO_coeff index)
        (alpha_1S_Ag,  coeffs_1S_Ag,  0, 0, 0, R_Ag, 0),
        (alpha_2SP_Ag, coeffs_2S_Ag,  0, 0, 0, R_Ag, 1),
        (alpha_2SP_Ag, coeffs_2P_Ag,  1, 0, 0, R_Ag, 2),
        (alpha_2SP_Ag, coeffs_2P_Ag,  0, 1, 0, R_Ag, 3),
        (alpha_2SP_Ag, coeffs_2P_Ag,  0, 0, 1, R_Ag, 4),
        (alpha_3SP_Ag, coeffs_3S_Ag,  0, 0, 0, R_Ag, 5),
        (alpha_3SP_Ag, coeffs_3P_Ag,  1, 0, 0, R_Ag, 6),
        (alpha_3SP_Ag, coeffs_3P_Ag,  0, 1, 0, R_Ag, 7),
        (alpha_3SP_Ag, coeffs_3P_Ag,  0, 0, 1, R_Ag, 8),
        (alpha_4D_Ag, coeffs_4D_Ag, 1, 1, 0, R_Ag, 9),
        (alpha_4D_Ag, coeffs_4D_Ag, 0, 1, 1, R_Ag, 10),
        #d z^2 done manually
        (alpha_4D_Ag, coeffs_4D_Ag, 1, 0, 1, R_Ag, 12),
        #d x^2-y^2 done manually
        (alpha_5SP_Ag, coeffs_5S_Ag,  0, 0, 0, R_Ag, 14),
        (alpha_5SP_Ag, coeffs_5P_Ag,  1, 0, 0, R_Ag, 15),
        (alpha_5SP_Ag, coeffs_5P_Ag,  0, 1, 0, R_Ag, 16),
        (alpha_5SP_Ag, coeffs_5P_Ag,  0, 0, 1, R_Ag, 17),
        (alpha_6D_Ag, coeffs_6D_Ag, 1, 1, 0, R_Ag, 18),
        (alpha_6D_Ag, coeffs_6D_Ag, 0, 1, 1, R_Ag, 19),
        #d z^2 done manually
        (alpha_6D_Ag, coeffs_6D_Ag, 1, 0, 1, R_Ag, 21),
        #d x^2 - y^2 done manually
        (alpha_7SP_Ag, coeffs_7S_Ag,  0, 0, 0, R_Ag, 23),
        (alpha_7SP_Ag, coeffs_7P_Ag,  1, 0, 0, R_Ag, 24),
        (alpha_7SP_Ag, coeffs_7P_Ag,  0, 1, 0, R_Ag, 25),
        (alpha_7SP_Ag, coeffs_7P_Ag,  0, 0, 1, R_Ag, 26),
        (alpha_1S_F,  coeffs_1S_F,  0, 0, 0, R_F, 27),
        (alpha_2SP_F, coeffs_2S_F,  0, 0, 0, R_F, 28),
        (alpha_2SP_F, coeffs_2P_F,  1, 0, 0, R_F, 29),
        (alpha_2SP_F, coeffs_2P_F,  0, 1, 0, R_F, 30),
        (alpha_2SP_F, coeffs_2P_F,  0, 0, 1, R_F, 31),
    ]

    DO = np.zeros_like(x, dtype=np.float64)
    for alphas, coeffs, a, b, c, center, i in basis_info:
        if DO_coeffs[i] != 0.0:
            ao = AO(alphas, coeffs, a, b, c, center, x, y, z)
            DO += DO_coeffs[i] * ao

    # Handle weird d orbitals manually
    # d_z2 = 1/2 * [2zz - xx - yy]
    if DO_coeffs[11] != 0.0:
        ao_zz = AO(alpha_4D_Ag, coeffs_4D_Ag, 0, 0, 2, R_Ag, x, y, z)
        ao_xx = AO(alpha_4D_Ag, coeffs_4D_Ag, 2, 0, 0, R_Ag, x, y, z)
        ao_yy = AO(alpha_4D_Ag, coeffs_4D_Ag, 0, 2, 0, R_Ag, x, y, z)
        DO += DO_coeffs[11] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    # d_x2_y2 = sqrt(3)/2 * (xx - yy)
    if DO_coeffs[13] != 0.0:
        ao_xx = AO(alpha_4D_Ag, coeffs_4D_Ag, 2, 0, 0, R_Ag, x, y, z)
        ao_yy = AO(alpha_4D_Ag, coeffs_4D_Ag, 0, 2, 0, R_Ag, x, y, z)
        DO += DO_coeffs[13] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

    if DO_coeffs[20] != 0.0:
        ao_zz = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 0, 2, R_Ag, x, y, z)
        ao_xx = AO(alpha_6D_Ag, coeffs_6D_Ag, 2, 0, 0, R_Ag, x, y, z)
        ao_yy = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 2, 0, R_Ag, x, y, z)
        DO += DO_coeffs[20] * 0.5 * (2 * ao_zz - ao_xx - ao_yy)

    if DO_coeffs[22] != 0.0:
        ao_xx = AO(alpha_6D_Ag, coeffs_6D_Ag, 2, 0, 0, R_Ag, x, y, z)
        ao_yy = AO(alpha_6D_Ag, coeffs_6D_Ag, 0, 2, 0, R_Ag, x, y, z)
        DO += DO_coeffs[22] * (np.sqrt(3)/2) * (ao_xx - ao_yy)

    # Normalize final DO
    DO /= np.sqrt(integrate_3d(np.abs(DO)**2, dV))
    return DO

# ---------------------------------------------
# Build rotation matrix from Euler angles (ZYZ convention)
# ---------------------------------------------

# Rotation matrix function (ZYZ convention)
def rotation_matrix(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz1 = np.array([[ ca, -sa, 0], [ sa,  ca, 0], [  0,   0, 1]])
    Ry  = np.array([[ cb,   0, sb], [  0,   1,  0], [-sb,  0, cb]])
    Rz2 = np.array([[ cg, -sg, 0], [ sg,  cg, 0], [  0,   0, 1]])
    return Rz1 @ Ry @ Rz2

# Rotate the grid
def rotate_grid(X, Y, Z, alpha, beta, gamma):
    R = rotation_matrix(alpha, beta, gamma)
    shape = X.shape
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)
    rotated_coords = R @ coords
    X_rot = rotated_coords[0].reshape(shape)
    Y_rot = rotated_coords[1].reshape(shape)
    Z_rot = rotated_coords[2].reshape(shape)
    return X_rot, Y_rot, Z_rot

# ---------------------------------------------
# Plot
# ---------------------------------------------

# Plotting function
def plot_DO(DO, R_A, R_B, title="Dyson Orbital Representation"):
    max_val = np.max(np.abs(DO))
    iso_positive = 0.1 * max_val
    iso_negative = -0.1 * max_val if np.min(DO) < 0 else None
    if iso_negative is not None and iso_negative < np.min(DO):
        iso_negative = None

    verts_pos, faces_pos, _, _ = marching_cubes(DO, level=iso_positive)
    scale = x[1] - x[0]
    verts_pos = verts_pos * scale + np.array([x[0], y[0], z[0]])

    verts_neg, faces_neg = None, None
    if iso_negative is not None:
        verts_neg, faces_neg, _, _ = marching_cubes(DO, level=iso_negative)
        verts_neg = verts_neg * scale + np.array([x[0], y[0], z[0]])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts_pos[:, 0], verts_pos[:, 1], verts_pos[:, 2],
                    triangles=faces_pos, color='royalblue', alpha=0.6)
    if verts_neg is not None:
        ax.plot_trisurf(verts_neg[:, 0], verts_neg[:, 1], verts_neg[:, 2],
                        triangles=faces_neg, color='red', alpha=0.6)

    ax.scatter(*R_A, color='black', s=100, label='Atom A')
    ax.scatter(*R_B, color='black', s=100, label='Atom B')
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min(), z.max()])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Build the Dyson Orbital
DO = build_DO(DO_coeffs_AgF_L, X, Y, Z, R_A, R_B)

# Euler angles
alpha, beta, gamma = 0, np.pi, 0

R = rotation_matrix(alpha, beta, gamma)

X_rot, Y_rot, Z_rot = rotate_grid(X, Y, Z, alpha, beta, gamma)

R_A_rot = R @ R_A.T
R_B_rot = R @ R_B.T

# Generate rotated DOs
DO_rot = build_DO(DO_coeffs_AgF_L, X, Y, Z, R_A_rot, R_B_rot)


# Plot original and rotated DOs
plot_DO(DO, R_A, R_B, title="Original Dyson Orbital")
plot_DO(DO_rot, R_A_rot, R_B_rot, title="After Rotation")

