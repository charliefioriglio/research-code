import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv
from scipy.interpolate import RegularGridInterpolator
from compute_phi import compute_Phi_m
from compute_angular import solve_angular_equation
from compute_radial import compute_radial_function
from compute_single_wavefunction import build_single_mode_wavefunction

def compute_total_wavefunction(m_max, n_max, E, a, D, L_max, xi_vals, eta_vals, phi_vals, theta_k, phi_k, R_max):

    xi_grid, eta_grid, phi_grid = np.meshgrid(xi_vals, eta_vals, phi_vals, indexing='ij')
    Psi_total = np.zeros_like(xi_grid, dtype=np.complex128)

    for m in range(-m_max, m_max + 1):

        for n in range(0, n_max + 1):
            Ψ_mn, _, _, _, _, _ = build_single_mode_wavefunction(
                m, n, E, a, D, L_max, xi_vals, eta_vals, phi_vals, theta_k, phi_k, R_max
            )
         
            # Add contribution
            Psi_total += Ψ_mn

    return Psi_total

def plot_wavefunction_slice(Psi, xi_vals, eta_vals, phi_vals, phi_index=None):
    if phi_index is None:
        phi_index = len(phi_vals) // 2

    Psi_slice = Psi[:, :, phi_index]
    density = np.abs(Psi_slice)**2
    xi_grid, eta_grid = np.meshgrid(xi_vals, eta_vals, indexing='ij')

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(eta_grid, xi_grid, np.log10(density + 1e-16), shading='auto', cmap='viridis')
    plt.colorbar(label=r'log$_{10}|\Psi|^2$')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\xi$')
    plt.title(rf'Total Ψ at φ = {phi_vals[phi_index]:.2f} rad')
    plt.tight_layout()
    plt.show()

# === Parameters ===
xi_vals = np.linspace(1.0, 200.0, 300)
eta_vals = np.linspace(-1.0, 1.0, 200)
phi_vals = np.linspace(0, 2 * np.pi, 240)

E = 0.1
a = 1
D = 0.0
L_max = 70
R_max = 70
m_max = 0
n_max = 40

# === Run and plot ===
theta_k = 0
phi_k = 0

Psi_total = compute_total_wavefunction(m_max, n_max, E, a, D, L_max, xi_vals, eta_vals, phi_vals, theta_k, phi_k, R_max)

plot_wavefunction_slice(Psi_total, xi_vals, eta_vals, phi_vals)

def cartesian_to_prolate_spheroidal(x, y, z, a):
    r1 = np.sqrt(x**2 + y**2 + (z - a)**2)
    r2 = np.sqrt(x**2 + y**2 + (z + a)**2)
    xi = (r1 + r2) / (2 * a)
    eta = (r1 - r2) / (2 * a)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    return xi, eta, phi

def plot_wavefunction_cartesian_slice(Psi, xi_vals, eta_vals, phi_vals, a, x_fixed, ylim, zlim, N):
    # Create Cartesian grid at fixed x
    y = np.linspace(ylim[0], ylim[1], N)
    z = np.linspace(zlim[0], zlim[1], N)
    Y, Z = np.meshgrid(y, z)
    X = np.full_like(Y, x_fixed)

    # Convert to prolate spheroidal coordinates
    xi, eta, phi = cartesian_to_prolate_spheroidal(X, Y, Z, a)

    # === Interpolator with periodic φ ===
    interp_func = RegularGridInterpolator((xi_vals, eta_vals, phi_vals), Psi, bounds_error=False, fill_value=0)

    # Stack points for interpolation
    pts = np.stack([xi.ravel(), eta.ravel(), phi.ravel()], axis=-1)

    # Mask points outside valid ranges
    mask = (xi.ravel() >= 1) & (np.abs(eta.ravel()) <= 1)
    Psi_vals = np.zeros_like(xi.ravel(), dtype=np.complex128)
    Psi_vals[mask] = interp_func(pts[mask])

    Psi_vals = Psi_vals.reshape(X.shape)

    # Plot |Ψ|²
    plt.figure(figsize=(7, 6))
    plt.pcolormesh(Y, Z, np.abs(Psi_vals)**2, shading='auto', cmap='viridis')
    plt.colorbar(label=r'$|\Psi(x,y,z={:.2f})|^2$'.format(x_fixed))
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title(f'Wavefunction magnitude squared slice through YZ plane')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# Suppose Psi_total is your full 3D wavefunction on (xi_vals, eta_vals, phi_vals) grids
plot_wavefunction_cartesian_slice(Psi_total, xi_vals, eta_vals, phi_vals, a,
                                 x_fixed=0.0, ylim=(-100, 100), zlim=(-100, 100), N=200)

