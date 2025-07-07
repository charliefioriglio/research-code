import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from math import factorial
from skimage.measure import marching_cubes

# Parameters
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

def build_DO(DO_coeffs, x, y, z, R_A, R_B):
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

# Build the Dyson Orbital (b3g or b2g)
DO = build_DO(DO_coeffs_b3g, X, Y, Z, R_A, R_B)

# Euler angles
alpha, beta, gamma = 0, np.pi / 2, 0

R = rotation_matrix(alpha, beta, gamma)

X_rot, Y_rot, Z_rot = rotate_grid(X, Y, Z, alpha, beta, gamma)

R_A_rot = R @ R_A.T
R_B_rot = R @ R_B.T

# Generate rotated DOs
DO_rot = build_DO(DO_coeffs_b3g, X, Y, Z, R_A_rot, R_B_rot)


# Plot original and rotated DOs
plot_DO(DO, R_A, R_B, title="Original Dyson Orbital")
plot_DO(DO_rot, R_A_rot, R_B_rot, title="After Rotation")

