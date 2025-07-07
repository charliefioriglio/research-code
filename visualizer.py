import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes


# ---------------------------------------------
# Define Grid
# ---------------------------------------------
L = 10
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
    return np.sum(np.abs(f)**2) * dV

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


DO = build_DO(DO_coeffs_b3g)
DO_norm = integrate_3d(DO, dV)
DO_renormalized = DO / np.sqrt(DO_norm)


print(f"Norm of DO: {DO_norm:.6f}")
print(f"Check Renomalization: {integrate_3d(DO_renormalized, dV):.6f}")


# ---------------------------------------------
# Plot
# ---------------------------------------------
# Define isovalues
max_val = np.max(np.abs(DO_renormalized))  # Absolute max ensures handling of only-positive cases
iso_positive = 0.01 * max_val

# Check if there are negative values and adjust iso_negative accordingly
iso_negative = -0.01 * max_val if np.min(DO_renormalized) < 0 else None  # Only define if negatives exist

# Ensure the negative isosurface level is within the range of the data
if iso_negative is not None and iso_negative < np.min(DO_renormalized):
    iso_negative = None  # Remove negative isosurface if it is out of bounds

# Extract positive isosurface
verts_pos, faces_pos, _, _ = marching_cubes(DO_renormalized, level=iso_positive)

# Convert indices to real coordinates
scale = (x[1] - x[0])  # Grid spacing
verts_pos = verts_pos * scale + np.array([x[0], y[0], z[0]])  # Positive isosurface

# Extract negative isosurface only if it exists and is within range
verts_neg = None
faces_neg = None
if iso_negative is not None:
    verts_neg, faces_neg, _, _ = marching_cubes(DO_renormalized, level=iso_negative)
    verts_neg = verts_neg * scale + np.array([x[0], y[0], z[0]])  # Negative isosurface

# Plot 3D isosurfaces
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# **Plot positive isosurface (blue)**
ax.plot_trisurf(verts_pos[:, 0], verts_pos[:, 1], verts_pos[:, 2], 
                triangles=faces_pos, color='royalblue', alpha=0.6)

# **Plot negative isosurface (red) only if it exists**
if verts_neg is not None:
    ax.plot_trisurf(verts_neg[:, 0], verts_neg[:, 1], verts_neg[:, 2], 
                    triangles=faces_neg, color='red', alpha=0.6)

# Mark atom positions
ax.scatter(*R_A, color='black', s=100, label='Atom A')
ax.scatter(*R_B, color='black', s=100, label='Atom B')

# Set axis limits to match grid range
ax.set_xlim([x.min(), x.max()])
ax.set_ylim([y.min(), y.max()])
ax.set_zlim([z.min(), z.max()])

# Ensure equal aspect ratio
ax.set_box_aspect([1, 1, 1])  

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Molecular Orbital Representation")
ax.legend()
plt.show()
