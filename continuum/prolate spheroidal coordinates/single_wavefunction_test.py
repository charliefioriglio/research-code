import numpy as np
import matplotlib.pyplot as plt
from compute_angular import build_analytic_angular_hamiltonian
from compute_radial import compute_radial_function
from compute_phi import compute_Phi_m
from scipy.special import lpmv

def build_single_mode_wavefunction(m, n, E, a, D, L_max, xi_vals, eta_vals, phi_vals, R_max=50):
    m_abs = abs(m)
    c = np.sqrt(2 * E * a**2)
    xi_grid, eta_grid, phi_grid = np.meshgrid(xi_vals, eta_vals, phi_vals, indexing='ij')

    eigvals_ang, eigvecs_ang, ell_vals = build_analytic_angular_hamiltonian(m_abs, L_max, E, a, D)
    v_ang = eigvecs_ang[:, n]

    # Angular function T_{m,n}(η)
    P_basis = np.array([lpmv(m_abs, l, eta_vals) for l in ell_vals])  # (N_basis, len(eta))
    T_eta = P_basis.T @ v_ang  # shape (len(eta),)
    if m < 0:
        T_eta *= (-1)**abs(m)

    # Radial function S_{m,n}(ξ)
    _, S_xi, _, _, _ = compute_radial_function(m, n, eigvals_ang, c, R_max=R_max, xi_vals=xi_vals)

    # Azimuthal function Φ_m(φ)
    Φ_m = compute_Phi_m(m, phi_vals)  # Use signed m here for phase correctness

    # Outer product to build full Ψ
    Ψ_mn = np.outer(S_xi, T_eta)[:, :, np.newaxis] * Φ_m[np.newaxis, np.newaxis, :]

    # Jacobian for full wavefunction norm
    jacobian = a**3 * (xi_grid**2 - eta_grid**2)
    dξ = xi_vals[1] - xi_vals[0]
    dη = eta_vals[1] - eta_vals[0]
    dφ = phi_vals[1] - phi_vals[0]
    norm = np.sum(np.abs(Ψ_mn)**2 * jacobian) * dξ * dη * dφ

    # --- Diagnostics: norm of individual components ---

    # Radial norm (weight = a^3 * (xi^2 - 1))
    radial_weight = a**3 * (xi_vals**2 - 1)
    norm_radial = np.sum(np.abs(S_xi)**2 * radial_weight) * dξ

    # Angular norm (uniform weight in eta, since basis orthonormal under S)
    norm_angular = np.sum(np.abs(T_eta)**2) * dη

    # Azimuthal norm (integral over phi)
    norm_phi = np.sum(np.abs(Φ_m)**2) * dφ

    print(f"Norm of Ψ_{{m={m}, n={n}}} = {norm:.6e}")
    print(f"Radial norm = {norm_radial:.6e}")
    print(f"Angular norm = {norm_angular:.6e}")
    print(f"Azimuthal norm = {norm_phi:.6e}")

    return Ψ_mn, S_xi, T_eta, Φ_m, norm

def plot_wavefunction_diagnostics(m, n, Ψ_mn, S_xi, T_eta, Φ_m, xi_vals, eta_vals, phi_vals, norm):

    print(f"Norm of Ψ_{{m={m}, n={n}}} = {norm:.6e}")

    φ_index = len(phi_vals) // 2
    Ψ_slice = Ψ_mn[:, :, φ_index]
    density = np.abs(Ψ_slice)**2
    xi_grid, eta_grid = np.meshgrid(xi_vals, eta_vals, indexing='ij')

    plt.figure(figsize=(6, 4))
    plt.pcolormesh(eta_grid, xi_grid, np.log10(density + 1e-16), shading='auto', cmap='viridis')
    plt.colorbar(label=r"log$_{10}$ |Ψ|²")
    plt.xlabel(r'η')
    plt.ylabel(r'ξ')
    plt.title(rf"Ψₘₙ Slice at φ = {phi_vals[φ_index]:.2f}, m={m}, n={n}")
    plt.tight_layout()
    plt.show()

    # Plot S_{m,n}(ξ)
    plt.figure()
    plt.plot(xi_vals, S_xi.real, label='Re S(ξ)')
    plt.plot(xi_vals, S_xi.imag, label='Im S(ξ)', linestyle='--')
    plt.xlabel(r'ξ')
    plt.ylabel(r'S(ξ)')
    plt.title(rf'Radial Function S_{{m={m}, n={n}}}(ξ)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot T_{m,n}(η)
    plt.figure()
    plt.plot(eta_vals, T_eta.real, label='Re T(η)')
    plt.plot(eta_vals, T_eta.imag, label='Im T(η)', linestyle='--')
    plt.xlabel(r'η')
    plt.ylabel(r'T(η)')
    plt.title(rf'Angular Function T_{{m={m}, n={n}}}(η)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot |Φ_m(φ)|
    plt.figure()
    plt.plot(phi_vals, np.real(Φ_m), label='Re Φₘ')
    plt.plot(phi_vals, np.imag(Φ_m), label='Im Φₘ', linestyle='--')
    plt.xlabel(r'φ')
    plt.ylabel(r'Φₘ(φ)')
    plt.title(rf'Azimuthal Function Φₘ(φ) for m = {m}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Grids
xi_vals = np.linspace(1.0, 50.0, 300)
eta_vals = np.linspace(-1.0, 1.0, 200)
phi_vals = np.linspace(0, 2 * np.pi, 120)

# Parameters
E = 0.1
a = 1.0
D = 0.0
L_max = 40
R_max = 40

# Mode
m = 0
n = 20

# Build and plot
Ψ_mn, S_xi, T_eta, Φ_m, norm = build_single_mode_wavefunction(m, n, E, a, D, L_max, xi_vals, eta_vals, phi_vals, R_max)
plot_wavefunction_diagnostics(m, n, Ψ_mn, S_xi, T_eta, Φ_m, xi_vals, eta_vals, phi_vals, norm)
