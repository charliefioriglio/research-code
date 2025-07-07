import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
from scipy.special import spherical_jn, sph_harm, factorial
import scipy.special as sp  # For spherical harmonics
import seaborn as sns
from scipy.special import gamma
from skimage.measure import marching_cubes
from scipy.special import genlaguerre, factorial, sph_harm
from scipy.integrate import simpson
import matplotlib.pyplot as plt


# Define spatial domain (3D grid)
L=40
x = np.linspace(-L, L, 100)
y = np.linspace(-L, L, 100)
z = np.linspace(-L, L, 100)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
R_A = np.array([0, 0, 0])
# Norm Procedure
def norm(Psi):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    volume_element = dx * dy * dz
    integral_value = integrate.simps(integrate.simps(integrate.simps(Psi**2, x), y), z)
    return integral_value

# Hydrogenic orbital ψ_{nlm}
def orb(n, l, m, x, y, z, Z=1):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    rho = 2 * Z * r / n

    norm_radial = np.sqrt((2 * Z / n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    L = genlaguerre(n - l - 1, 2 * l + 1)
    R_nl = norm_radial * np.exp(-rho / 2) * rho**l * L(rho)

    Y_lm = sph_harm(m, l, phi, theta)
    return R_nl * Y_lm

psi_molecule = orb(3, 2, -1, X, Y, Z)

# Check Normalization
DO_norm = norm(psi_molecule)
print(f"Norm: {DO_norm:.6f}")

# Plot
# Define isovalues
max_val = np.max(np.abs(psi_molecule))  # Absolute max ensures handling of only-positive cases
iso_positive = 0.1 * max_val

# Check if there are negative values and adjust iso_negative accordingly
iso_negative = -0.1 * max_val if np.min(psi_molecule) < 0 else None  # Only define if negatives exist

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