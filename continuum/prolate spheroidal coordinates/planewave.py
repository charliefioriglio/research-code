import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, sph_harm

def compute_spherical_plane_wave(k_mag, theta_k, phi_k, X, Y, Z, L_max):
    """
    Computes a continuum-normalized plane wave via its spherical wave expansion.

    Parameters:
    - k_mag: magnitude of the wavevector (in a.u.)
    - theta_k, phi_k: angles of the wavevector (in radians)
    - X, Y, Z: Cartesian grid
    - L_max: truncation of spherical wave expansion

    Returns:
    - Psi: complex-valued wavefunction Ψ_k(r)
    """
    # Grid to spherical coords
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(np.clip(Z / (r + 1e-20), -1.0, 1.0))  # add epsilon to avoid divide by zero
    phi = np.arctan2(Y, X)

    Psi = np.zeros_like(r, dtype=complex)

    for l in range(L_max + 1):
        jl = spherical_jn(l, k_mag * r)  # spherical Bessel
        for m in range(-l, l + 1):
            Ylm_r = sph_harm(m, l, phi, theta)  # Ylm(r)
            Ylm_k = sph_harm(m, l, phi_k, theta_k)  # Ylm(k)
            term = 1j**l * jl * Ylm_r * np.conj(Ylm_k)
            Psi += term

    # Normalization factor (from plane wave expansion)
    Psi *= (4 * np.pi) / (2 * np.pi)**(3/2)
    return Psi

# Define 3D grid
L = 30
n_pts = 100
x = np.linspace(-L, L, n_pts)
y = np.linspace(-L, L, n_pts)
z = np.linspace(-L, L, n_pts)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Wavevector
E_eV = 1
hartree = 27.2114
E_au = E_eV / hartree
k_mag = np.sqrt(2 * E_au)
theta_k = 0
phi_k = 0

# Compute wavefunction
Psi_k = compute_spherical_plane_wave(k_mag, theta_k, phi_k, X, Y, Z, 0)

# Plot xz slice
def plot_xz_slice(Psi, X, Z, Y, y_index=None, title=''):
    if y_index is None:
        y_index = Psi.shape[1] // 2
    Psi_slice = np.real(Psi[:, y_index, :])
    X_slice = X[:, y_index, :]
    Z_slice = Z[:, y_index, :]

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X_slice, Z_slice, Psi_slice, shading='auto', cmap='plasma')
    plt.xlabel('x (Bohr)')
    plt.ylabel('z (Bohr)')
    plt.title(title)
    plt.colorbar(label='Re[Ψ]')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

plot_xz_slice(Psi_k, X, Z, Y, title='Spherical Expansion: k || z')

def plot_plane_wave_in_prolate_coords(Psi_func, a, k_mag, theta_k, phi_k, xi_vals, eta_vals, phi_fixed):
    """
    Evaluate and plot Ψ_k(r) in (ξ, η) coordinates at fixed φ.
    Uses Cartesian mapping from prolate spheroidal grid.
    """
    xi_grid, eta_grid = np.meshgrid(xi_vals, eta_vals, indexing='ij')
    phi_grid = np.full_like(xi_grid, phi_fixed)

    # Convert (ξ, η, φ) to Cartesian (x, y, z)
    x = a * np.sqrt((xi_grid**2 - 1) * (1 - eta_grid**2)) * np.cos(phi_grid)
    y = a * np.sqrt((xi_grid**2 - 1) * (1 - eta_grid**2)) * np.sin(phi_grid)
    z = a * xi_grid * eta_grid

    # Evaluate Ψ at these (x, y, z)
    Psi_vals = Psi_func(k_mag, theta_k, phi_k, x, y, z, L_max=40)

    # Plot |Ψ|²
    plt.figure(figsize=(7, 5))
    plt.pcolormesh(eta_grid, xi_grid, np.log10(np.abs(Psi_vals)**2 + 1e-16), shading='auto', cmap='viridis')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\xi$')
    plt.title(fr'$|\Psi_k(\xi,\eta)|^2$ at φ = {phi_fixed:.2f} rad')
    plt.colorbar(label=r'$|\Psi|^2$')
    plt.tight_layout()
    plt.show()


# Prolate coordinates
a = 1.0
xi_vals = np.linspace(1.0, 200.0, 500)
eta_vals = np.linspace(-1.0, 1.0, 250)
phi_fixed = 0.0  # Can vary this if desired

plot_plane_wave_in_prolate_coords(
    compute_spherical_plane_wave,
    a,
    k_mag,
    theta_k,
    phi_k,
    xi_vals,
    eta_vals,
    phi_fixed
)
def plot_prolate_wavefunction_slice(
    wavefunction_func,
    m_max, n_max, E, a, D, L_max,
    xi_vals, eta_vals, phi_fixed,
    theta_k, phi_k,
    R_max=50
):
    """
    Plots a fixed-φ slice of a prolate spheroidal wavefunction Ψ(ξ, η, φ).

    Parameters:
        wavefunction_func: callable returning Ψ(ξ, η, φ) on the grid
        all other args: as used in compute_total_wavefunction
    """
    phi_vals = np.linspace(0, 2 * np.pi, 100)  # for building full wavefunction
    xi_grid, eta_grid = np.meshgrid(xi_vals, eta_vals, indexing='ij')
    phi_grid = np.full_like(xi_grid, phi_fixed)

    # Evaluate full 3D Ψ
    Psi_3d = wavefunction_func(
        m_max, n_max, E, a, D, L_max,
        xi_vals, eta_vals, phi_vals,
        theta_k, phi_k,
        R_max=R_max
    )

    # Find closest φ index
    phi_index = np.argmin(np.abs(phi_vals - phi_fixed))
    Psi_slice = Psi_3d[:, :, phi_index]

    # Plot log |Ψ|² in (ξ, η)
    plt.figure(figsize=(7, 5))
    plt.pcolormesh(
        eta_grid, xi_grid,
        np.log10(np.abs(Psi_slice)**2 + 1e-16),
        shading='auto', cmap='viridis'
    )
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\xi$')
    plt.title(fr'$|\Psi(\xi,\eta)|^2$ at φ = {phi_fixed:.2f} rad')
    plt.colorbar(label=r'$\log_{10} |\Psi|^2$')
    plt.tight_layout()
    plt.show()
# Parameters
a = 1.0
E = 1 / 27.2
D = 0.0
m_max = 0
n_max = 60
L_max = 60
theta_k = 0.0
phi_k = 0.0
phi_fixed = 0.0

xi_vals = np.linspace(1.0, 200.0, 400)
eta_vals = np.linspace(-1.0, 1.0, 200)

plot_prolate_wavefunction_slice(
    compute_total_wavefunction,
    m_max, n_max, E, a, D, L_max,
    xi_vals, eta_vals, phi_fixed,
    theta_k, phi_k
)

