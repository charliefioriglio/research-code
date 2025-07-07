import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, spherical_jn


def build_single_mode_spherical_wavefunction(m, l, k, r_vals, theta_vals, phi_vals):
    """
    Construct a single spherical harmonic mode of a continuum-normalized plane wave.
    
    Parameters:
    - m: azimuthal quantum number
    - l: orbital angular momentum quantum number
    - k: wavevector magnitude
    - r_vals, theta_vals, phi_vals: 1D arrays for spherical coordinates

    Returns:
    - Psi_lm: complex-valued Ψ_{l,m}(r,θ,φ)
    - R_l: radial function
    - Y_lm: angular spherical harmonic
    - Phi_m: azimuthal phase
    """
    r_grid, theta_grid, phi_grid = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')

    # Radial function: spherical Bessel function
    R_l = spherical_jn(l, k * r_vals)  # real-valued

    # Angular function: spherical harmonic (only theta dependence here)
    Y_lm = sph_harm(m, l, 0, theta_vals)  # phi=0 so Y_lm(θ,φ) = Θ_lm(θ) * exp(imφ) |φ-factored out|

    # Azimuthal function:
    Phi_m = np.exp(1j * m * phi_vals) / np.sqrt(2 * np.pi)

    # Full Ψ_{l,m}(r,θ,φ) = j_l(kr) * Y_lm(θ,0) * exp(i m φ)
    Psi_lm = (
        R_l[:, np.newaxis, np.newaxis]
        * Y_lm[np.newaxis, :, np.newaxis]
        * Phi_m[np.newaxis, np.newaxis, :]
    )

    return Psi_lm, R_l, Y_lm, Phi_m


def plot_spherical_components(r_vals, theta_vals, phi_vals, R_l, Y_lm, Phi_m, m, l):
    # Radial
    plt.figure()
    plt.plot(r_vals, R_l, label=f'j_{l}(kr)')
    plt.title(f"Radial Part j_{l}(kr)")
    plt.xlabel("r")
    plt.ylabel(f"j_{l}(kr)")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Angular (theta slice)
    plt.figure()
    plt.plot(theta_vals, Y_lm.real, label='Re Y')
    plt.plot(theta_vals, Y_lm.imag, label='Im Y', linestyle='--')
    plt.title(f"Angular Part Y_{l}^{m}(θ), φ=0")
    plt.xlabel("θ (rad)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Azimuthal
    plt.figure()
    plt.plot(phi_vals, Phi_m.real, label='Re Φ')
    plt.plot(phi_vals, Phi_m.imag, label='Im Φ', linestyle='--')
    plt.title(f"Azimuthal Part exp(i m φ) / √(2π), m={m}")
    plt.xlabel("φ (rad)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Parameters
l = 3
m = 2
E = 0.1
k = np.sqrt(2 * E)

# Grids
r_vals = np.linspace(0.1, 20.0, 300)
theta_vals = np.linspace(0, np.pi, 200)
phi_vals = np.linspace(0, 2 * np.pi, 120)

# Build and plot
Psi_lm, R_l, Y_lm, Phi_m = build_single_mode_spherical_wavefunction(m, l, k, r_vals, theta_vals, phi_vals)
plot_spherical_components(r_vals, theta_vals, phi_vals, R_l, Y_lm, Phi_m, m, l)

def prolate_to_spherical_coords(xi, eta, phi, a):
    """
    Convert from prolate spheroidal coordinates to spherical coordinates.
    Assumes the focal length 2a.

    Returns:
    - r, theta, phi (spherical coordinates)
    """
    # Cartesian from prolate
    x = a * np.sqrt((xi**2 - 1) * (1 - eta**2)) * np.cos(phi)
    y = a * np.sqrt((xi**2 - 1) * (1 - eta**2)) * np.sin(phi)
    z = a * xi * eta

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))

    return r, theta, phi


def plot_spherical_wave_in_prolate_coords(Psi_lm_func, l, m, k, a, xi_vals, eta_vals, phi_fixed):
    """
    Evaluate Ψ_{l,m} at (ξ, η, φ_fixed) using coordinate conversion.
    Plot |Ψ(ξ, η, φ_fixed)|².
    """
    xi_grid, eta_grid = np.meshgrid(xi_vals, eta_vals, indexing='ij')
    phi_grid = np.full_like(xi_grid, phi_fixed)

    # Convert to spherical coordinates
    r_grid, theta_grid, phi_grid = prolate_to_spherical_coords(xi_grid, eta_grid, phi_grid, a)

    # Evaluate components
    R_l_vals = spherical_jn(l, k * r_grid)
    Y_lm_vals = sph_harm(m, l, phi_grid, theta_grid)
    Psi_vals = R_l_vals * Y_lm_vals

    # Plot
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(eta_grid, xi_grid, np.log10(np.abs(Psi_vals)**2 + 1e-16), shading='auto', cmap='viridis')
    plt.colorbar(label=r'$|\Psi_{l,m}(\xi, \eta)|^2$')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\xi$')
    plt.title(rf'Spherical Mode $|\Psi_{{{l},{m}}}|^2$ in Prolate Coordinates at φ={phi_fixed:.2f}')
    plt.tight_layout()
    plt.show()


# Plot spherical Ψ_lm in prolate space
a = 1.0
xi_vals = np.linspace(1.0, 50.0, 200)
eta_vals = np.linspace(-1.0, 1.0, 200)
phi_fixed = 0.0

plot_spherical_wave_in_prolate_coords(Psi_lm, l, m, k, a, xi_vals, eta_vals, phi_fixed)
