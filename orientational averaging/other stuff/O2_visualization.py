import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
from scipy.special import spherical_jn, sph_harm, factorial
import scipy.special as sp  # For spherical harmonics
import seaborn as sns
from scipy.special import gamma
from skimage.measure import marching_cubes

# Atomic units

# Define Constants
bl_A = 1.35  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au

R_A = np.array([0, 0, (bl_au / 2)])
R_B = np.array([0, 0, -(bl_au / 2)])

# Define spatial domain (3D grid)
L=5
x = np.linspace(-L, L, 100)
y = np.linspace(-L, L, 100)
z = np.linspace(-L, L, 100)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Norm Procedure
def norm(Psi):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    volume_element = dx * dy * dz
    integral_value = integrate.simps(integrate.simps(integrate.simps(Psi**2, x), y), z)
    return integral_value

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
S1_A = S(C1s, E1, X, Y, Z, R_A) * DO_C[0]
S2_A = S(C2s, E2, X, Y, Z, R_A) * DO_C[1]
Px_A = Px(Cp, E2, X, Y, Z, R_A) * DO_C[2]
Py_A = Py(Cp, E2, X, Y, Z, R_A) * DO_C[3]
Pz_A = Pz(Cp, E2, X, Y, Z, R_A) * DO_C[4]
S1_B = S(C1s, E1, X, Y, Z, R_B) * DO_C[5]
S2_B = S(C2s, E2, X, Y, Z, R_B) * DO_C[6]
Px_B = Px(Cp, E2, X, Y, Z, R_B) * DO_C[7]
Py_B = Py(Cp, E2, X, Y, Z, R_B) * DO_C[8]
Pz_B = Pz(Cp, E2, X, Y, Z, R_B) * DO_C[9]

psi_molecule = S1_A + S2_A + Px_A + Py_A + Pz_A + S1_B + S2_B + Px_B + Py_B + Pz_B

# Check Normalization
DO_norm = norm(psi_molecule)
print(f"Norm: {DO_norm:.6f}")

# Plot
# Define isovalues
max_val = np.max(np.abs(psi_molecule))  # Absolute max ensures handling of only-positive cases
iso_positive = .1 * max_val

# Check if there are negative values and adjust iso_negative accordingly
iso_negative = -.1 * max_val if np.min(psi_molecule) < 0 else None  # Only define if negatives exist

# Ensure the negative isosurface level is within the range of the data
if iso_negative is not None and iso_negative < np.min(psi_molecule):
    iso_negative = None  # Remove negative isosurface if it is out of bounds

# Extract positive isosurface
verts_pos, faces_pos, _, _ = marching_cubes(psi_molecule, level=iso_positive)

# Convert indices to real coordinates
scale = (x[1] - x[0])  # Grid spacing
verts_pos = verts_pos * scale + np.array([x[0], y[0], z[0]])  # Positive isosurface

# Extract negative isosurface only if it exists and is within range
verts_neg = None
faces_neg = None
if iso_negative is not None:
    verts_neg, faces_neg, _, _ = marching_cubes(psi_molecule, level=iso_negative)
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