import numpy as np
from scipy.special import genlaguerre, factorial, sph_harm
from scipy.integrate import simpson
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

# Create 3D grid
L = 2.5
N = 100
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Plane-wave continuum function
def el(k, x, y, z):
    k_dot_r = k[0]*x + k[1]*y + k[2]*z
    return np.exp(1j * k_dot_r)

# O2 Dyson orbital
#Parameters
bl_A = 1.35  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au
R_A = np.array([0, 0, (bl_au / 2)])
R_B = np.array([0, 0, -(bl_au / 2)])

# Norm Procedure
def norm(f):
    norm = simpson(
        simpson(
            simpson(np.abs(f)**2, x=x, axis=0),  # integrate over x
            x=y, axis=0),                       # integrate over y
        x=z, axis=0)                            # integrate over z
    return norm

# Define Dyson Orbital

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

DO_C2 = np.array([-4.44110614e-17, 1.97490563e-16, -7.47775656e-1, -5.13099748e-16, -2.10346844e-16, 1.99423414e-16, -1.83487917e-16, 7.47775656e-1, 1.06160592e-15, 5.17773631e-16])

S1_A = S(C1s, E1, Y, X, Z, R_A) * DO_C[0]
S2_A = S(C2s, E2, Y, X, Z, R_A) * DO_C[1]
Px_A = Px(Cp, E2, Y, X, Z, R_A) * DO_C[2]
Py_A = Py(Cp, E2, Y, X, Z, R_A) * DO_C[3]
Pz_A = Pz(Cp, E2, Y, X, Z, R_A) * DO_C[4]
S1_B = S(C1s, E1, Y, X, Z, R_B) * DO_C[5]
S2_B = S(C2s, E2, Y, X, Z, R_B) * DO_C[6]
Px_B = Px(Cp, E2, Y, X, Z, R_B) * DO_C[7]
Py_B = Py(Cp, E2, Y, X, Z, R_B) * DO_C[8]
Pz_B = Pz(Cp, E2, Y, X, Z, R_B) * DO_C[9]

S1_A2 = S(C1s, E1, Y, X, Z, R_A) * DO_C2[0]
S2_A2 = S(C2s, E2, Y, X, Z, R_A) * DO_C2[1]
Px_A2 = Px(Cp, E2, Y, X, Z, R_A) * DO_C2[2]
Py_A2 = Py(Cp, E2, Y, X, Z, R_A) * DO_C2[3]
Pz_A2 = Pz(Cp, E2, Y, X, Z, R_A) * DO_C2[4]
S1_B2 = S(C1s, E1, Y, X, Z, R_B) * DO_C2[5]
S2_B2 = S(C2s, E2, Y, X, Z, R_B) * DO_C2[6]
Px_B2 = Px(Cp, E2, Y, X, Z, R_B) * DO_C2[7]
Py_B2 = Py(Cp, E2, Y, X, Z, R_B) * DO_C2[8]
Pz_B2 = Pz(Cp, E2, Y, X, Z, R_B) * DO_C2[9]

DO = S1_A + S2_A + Px_A + Py_A + Pz_A + S1_B + S2_B + Px_B + Py_B + Pz_B

DO2 = S1_A2 + S2_A2 + Px_A2 + Py_A2 + Pz_A2 + S1_B2 + S2_B2 + Px_B2 + Py_B2 + Pz_B2

# Dipole matrix element integrands
def integrand_par(k, x, y, z):
    k_hat = k / np.linalg.norm(k)
    psi_el = el(k, x, y, z)
    mu_dot_r = k_hat[0]*x + k_hat[1]*y + k_hat[2]*z
    return psi_el * mu_dot_r

def integrand_perp(k, k_el, x, y, z):
    k_hat = k / np.linalg.norm(k)
    psi_el = el(k_el, x, y, z)
    mu_dot_r = k_hat[0]*x + k_hat[1]*y + k_hat[2]*z
    return psi_el * mu_dot_r

# Simpson-based 3D integration
def integrate_3d_simps(f):
    val = simpson(
        simpson(
            simpson(f, x=x, axis=0),  # integrate over x
            x=y, axis=0),                       # integrate over y
        x=z, axis=0)                            # integrate over z
    return np.abs(val)**2


# Angle grid

# Constants for the grid size and Euler angles
nabc = 150  # Number of grid points

# Create a grid for the Euler angles
ang_grid = np.zeros((3, nabc))

ang_grid[0] = [1.07393, 3.31052, 1.74962, 1.64604, 6.10750, 1.75961, 5.53027, 5.52842, 2.28479, 3.58752, 1.66523, 0.84226, 3.29536, 5.57673, 2.47174, 0.92423, 3.93291, 3.31288, 1.90607, 5.84085, 0.38988, 3.04104, 3.73432, 0.06801, 4.53341, 3.95200, 4.40989, 3.00258, 0.72053, 3.05150,  3.43130, 4.27813, 0.33053, 0.21824, 1.26162, 2.18568, 4.36603, 0.72968, 0.09293, 1.39264, 0.61877, 3.83971, 1.33503, 2.72987, 5.64007, 0.64928, 4.61242, 3.37637, 5.84322, 3.88503, 3.30267, 2.80485, 5.96564, 3.63200, 3.21888, 5.15715, 6.16719, 5.35457, 0.54376, 4.22616, 4.85400, 3.31674, 1.37092, 4.28857, 1.13002, 0.71738, 0.54974, 0.55092, 1.24132, 3.64290, 1.64918, 0.21685, 3.08618, 0.92048, 2.12227, 2.48303, 5.11643, 4.98922, 5.85691, 5.20647, 0.55287, 4.78971, 0.20077, 1.62743, 1.16900, 5.35471, 3.02272, 6.06776, 2.18861, 4.82467, 5.11437, 2.76769, 1.87105, 4.61828, 4.49672, 3.57552, 2.38265, 2.76245, 4.72005, 4.04847, 4.67308, 0.03871, 5.47292, 0.39803, 5.04133, 1.91134, 2.47181, 2.18850, 0.02788, 1.64398, 2.99010, 5.39785, 2.22545, 0.13728, 4.95468, 3.95088, 4.48075, 3.89107, 5.28079, 1.04117, 1.52149, 6.18079, 5.73604, 1.91169, 5.92459, 5.51662, 3.78850, 2.46795, 4.85331, 2.43693, 0.84941, 4.11191, 1.05014, 3.60250, 2.66381, 5.14617, 1.60611, 5.85919, 6.20983, 3.03865, 5.79858, 4.28145, 2.75753, 4.16086, 5.81163, 4.79705, 1.95620, 2.56970, 1.91953, 1.30330]

ang_grid[1] = [2.37487, 2.01802, 0.19298, 1.42732, 0.94148, 2.87768, 0.43346, 2.23837, 2.38175, 1.86742, 0.48430, 1.47576, 2.33447, 1.61340, 1.85509, 2.01268, 0.75689, 1.06096, 1.28682, 2.45608, 1.22613, 0.85503, 2.48750, 2.59037, 1.12758, 1.08387, 1.71823, 2.89856, 1.19880, 1.82581, 0.72273, 2.03217, 2.31736, 1.47451, 2.64602, 1.75559, 0.58855, 1.75632, 1.74660, 1.57420, 0.64023, 1.67455, 1.85446, 1.02968, 1.28404, 2.56314, 1.94129, 0.21312, 0.13739, 1.37820, 1.37309, 2.42800, 1.21872, 2.18767, 2.62084, 2.30207, 2.00292, 1.40952, 1.48686, 1.22480, 2.13311, 1.69066, 1.25267, 2.52738, 1.45655, 2.25519, 2.00568, 2.85707, 2.12000, 0.96483, 1.75524, 2.03425, 0.51224, 0.89061, 0.67305, 1.24734, 0.87917, 0.58117, 1.50352, 2.01029, 0.93519, 1.36289, 0.97133, 0.77734, 0.64195, 2.56591, 1.19311, 1.72994, 1.16124, 2.76802, 1.23525, 1.94278, 2.56288, 0.31374, 2.25460, 1.54648, 0.91810, 1.35613, 1.64956, 2.27038, 0.80996, 0.41045, 0.74431, 1.74535, 1.53611, 1.60435, 1.55181, 1.45794, 1.21354, 1.09534, 2.15902, 1.07567, 2.06928, 0.69204, 1.83188, 2.77746, 1.42929, 1.99852, 1.71301, 1.17178, 2.35523, 1.46674, 0.97673, 2.24095, 0.67027, 1.93475, 0.47169, 2.66075, 1.07081, 0.42167, 0.36992, 1.78975, 1.74413, 1.25456, 0.70579, 3.06199, 2.05766, 2.12962, 2.29422, 1.51664, 1.83453, 0.90821, 1.65087, 1.50476, 2.78176, 2.47069, 0.94743, 2.17897, 1.92705, 0.94949]

ang_grid[2] = [0.00649606, 0.00677567, 0.00640311, 0.00666473, 0.00669922, 0.00663394, 0.00674343, 0.00667127, 0.00668776, 0.00676721, 0.00665191, 0.00663109, 0.00663458, 0.00674962, 0.00667875, 0.00676506, 0.00663678, 0.00664742, 0.00663493, 0.00677988, 0.00666949, 0.00678447, 0.00678442, 0.00674655, 0.00660888, 0.00662420, 0.00663710, 0.00638865, 0.00674299, 0.00667288, 0.00663664, 0.00663669, 0.00669460, 0.00674184, 0.00666331, 0.00668894, 0.00668811, 0.00678161, 0.00666844, 0.00649848, 0.00672334, 0.00663329, 0.00657182, 0.00678480, 0.00678987, 0.00668689, 0.00668108, 0.00665181, 0.00659894, 0.00675097, 0.00676891, 0.00677254, 0.00678260, 0.00672571, 0.00664667, 0.00640289, 0.00656302, 0.00671165, 0.00670240, 0.00655946, 0.00665147, 0.00671663, 0.00668897, 0.00670018, 0.00646668, 0.00646788, 0.00663172, 0.00672750, 0.00657072, 0.00637692, 0.00672365, 0.00674388, 0.00667000, 0.00666926, 0.00675168, 0.00664912, 0.00663779, 0.00677741, 0.00670043, 0.00660064, 0.00656175, 0.00666901, 0.00638311, 0.00668824, 0.00678113, 0.00665192, 0.00672800, 0.00638353, 0.00639049, 0.00675165, 0.00663724, 0.00639167, 0.00676403, 0.00675114, 0.00666951, 0.00662358, 0.00666779, 0.00663807, 0.00668906, 0.00678518, 0.00666891, 0.00666294, 0.00671233, 0.00670227, 0.00677799, 0.00676458, 0.00677420, 0.00666742, 0.00659765, 0.00672893, 0.00665960, 0.00677759, 0.00676543, 0.00657994, 0.00675244, 0.00666695, 0.00672720, 0.00664683, 0.00674483, 0.00669447, 0.00672329, 0.00659747, 0.00678974, 0.00678624, 0.00674990, 0.00666400, 0.00668276, 0.00666590, 0.00637544, 0.00664386, 0.00667255, 0.00637753, 0.00676400, 0.00663239, 0.00669928, 0.00664242, 0.00676106, 0.00672395, 0.00666835, 0.00677843, 0.00658002, 0.00672617, 0.00666076, 0.00662597, 0.00668821, 0.00664327, 0.00664268, 0.00667836, 0.00678659, 0.00674897]

# Compute transition dipole matrix

# Rotation matrices for Euler angles
def rotation_matrix(a, b, g):
    R = np.array([[np.cos(a) * np.cos(b) * np.cos(g) - np.sin(a) * np.sin(g), -np.cos(g) * np.sin(a) - np.cos(a) * np.cos(b) * np.sin(g), np.cos(a) * np.sin(b)],
                  [np.cos(a) * np.sin(g) + np.cos(b) * np.cos(g) * np.sin(a), np.cos(a) * np.cos(g) - np.cos(b) * np.sin(a) * np.sin(g), np.sin(a) * np.sin(b)],
                  [-np.cos(g) * np.sin(b), np.sin(b) * np.sin(g), np.cos(b)]])
    return R

# Euler rotation
def rotate_dyson_orbital(DO, X, Y, Z, R):
    """Rotate the Dyson orbital by applying R to the spatial grid."""
    # Flatten the spatial grids into 1D arrays
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])  # Shape (3, N)

    # Apply rotation: new_points = R @ old_points
    rotated_points = R @ points  # Shape (3, N)

    # Extract rotated coordinates
    X_rot, Y_rot, Z_rot = rotated_points.reshape(3, *X.shape)

    # Interpolate Dyson Orbital onto the new grid
    rotated_DO = map_coordinates(DO, [X_rot.ravel(), Y_rot.ravel(), Z_rot.ravel()], order=1, mode='nearest')

    # Reshape back into the original grid shape
    return rotated_DO.reshape(DO.shape)


def par_avg(DO, k, ang_grid, num_angles):
    """
    Perform the orientational averaging of the Dyson orbital over the Euler angles.
    """
    avg_DIFK = 0  # Initialize the averaged value

    # Loop over all Euler angle combinations
    for i in range(num_angles):
        alpha, beta, gamma = ang_grid[0][i], ang_grid[1][i], ang_grid[2][i]  # Extract angles from the grid

        # Get the rotation matrix
        R = rotation_matrix(alpha, beta, gamma)

        # Rotate the Dyson orbital (assuming this applies a proper transformation)
        rotated_DO = rotate_dyson_orbital(DO, X, Y, Z, R)
        
        # Calculate the integrand for this specific rotation
        integrand_rot = integrand_par(k, X, Y, Z) * rotated_DO
        
        # Integrate over the spatial grid (X, Y, Z)
        integrand_integrated = integrate_3d_simps(integrand_rot)
        
        # Accumulate the result for the average
        avg_DIFK += integrand_integrated

    return avg_DIFK


def perp_avg(DO, k, k_el, ang_grid, num_angles):
    """
    Perform the orientational averaging of the Dyson orbital over the Euler angles.
    """
    avg_DIFK = 0  # Initialize the averaged value

    # Loop over all Euler angle combinations
    for i in range(num_angles):
        alpha, beta, gamma = ang_grid[0][i], ang_grid[1][i], ang_grid[2][i]  # Extract angles from the grid

        # Get the rotation matrix
        R = rotation_matrix(alpha, beta, gamma)

        # Rotate the Dyson orbital (assuming this applies a proper transformation)
        rotated_DO = rotate_dyson_orbital(DO, X, Y, Z, R)
        
        # Calculate the integrand for this specific rotation
        integrand_rot = integrand_perp(k, k_el, X, Y, Z) * rotated_DO
        
        # Integrate over the spatial grid (X, Y, Z)
        integrand_integrated = integrate_3d_simps(integrand_rot)
        
        # Accumulate the result for the average
        avg_DIFK += integrand_integrated

    return avg_DIFK


# Parameters
E_eV = np.linspace(0.01, 8, 5)  # energy in eV
hartree = 27.2114
beta_vals = []

for E in E_eV:
    E_au = E / hartree
    k_mag = np.sqrt(2 * E_au)

    kx = k_mag * np.array([1, 0, 0])
    ky = k_mag * np.array([0, 1, 0])
    kz = k_mag * np.array([0, 0, 1])

    D_par = 0
    D_perp = 0
    # Parallel terms
    D_par += (
             par_avg(DO, kz, ang_grid, nabc) +
             par_avg(DO2, kz, ang_grid, nabc)
    )

    # Perpendicular terms
    D_perp += 1/2 * (
        perp_avg(DO, kz, kx, ang_grid, nabc) +
        perp_avg(DO, kz, ky, ang_grid, nabc) +
        perp_avg(DO2, kz, kx, ang_grid, nabc) +
        perp_avg(DO2, kz, ky, ang_grid, nabc)
    )

    beta = 2 * (D_par - D_perp) / (D_par + 2 * D_perp)
    beta_vals.append(beta)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(E_eV, beta_vals, marker='o')
plt.xlabel('Photoelectron Kinetic Energy (eV)')
plt.ylabel(r'$\beta$ parameter')
plt.title(r'$\beta$ vs. Kinetic Energy')
plt.ylim(-1, 2)
plt.grid(True)
plt.tight_layout()
plt.show()