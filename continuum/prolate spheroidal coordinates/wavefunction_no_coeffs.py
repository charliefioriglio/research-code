import numpy as np
import matplotlib.pyplot as plt
from compute_phi import compute_Phi_m
from compute_angular import solve_angular_equation
from compute_radial import compute_radial_function
from compute_single_wavefunction import build_single_mode_wavefunction

def compute_total_wavefunction(m_max, n_max, E, a, D, L_max, xi_vals, eta_vals, phi_vals, R_max=50):
    xi_grid, eta_grid, phi_grid = np.meshgrid(xi_vals, eta_vals, phi_vals, indexing='ij')
    Psi_total = np.zeros_like(xi_grid, dtype=np.complex128)

    for m in range(-m_max, m_max + 1):
        m_abs = abs(m)

        # Get angular eigenvectors once to determine how many n values are allowed
        eigvals_ang, eigvecs_ang, _ = solve_angular_equation(m_abs, L_max, E, a, D)
        n_limit = min(n_max, eigvecs_ang.shape[1] - 1)  # indexable limit

        for n in range(m_abs, n_limit + 1):
            Ψ_mn, _, _, _ = build_single_mode_wavefunction(m, n, E, a, D, L_max, xi_vals, eta_vals, phi_vals, R_max)
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

def compute_norm(Psi, xi_vals, eta_vals, phi_vals, a):
    xi_grid, eta_grid, _ = np.meshgrid(xi_vals, eta_vals, phi_vals, indexing='ij')
    jacobian = a**3 * (xi_grid**2 - eta_grid**2)
    dξ = xi_vals[1] - xi_vals[0]
    dη = eta_vals[1] - eta_vals[0]
    dφ = phi_vals[1] - phi_vals[0]
    norm = np.sum(np.abs(Psi)**2 * jacobian) * dξ * dη * dφ
    return norm

# === Parameters ===
xi_vals = np.linspace(1.0, 30.0, 300)
eta_vals = np.linspace(-1.0, 1.0, 200)
phi_vals = np.linspace(0, 2 * np.pi, 120)

E = 0.1
a = 1.0
D = 0.0
L_max = 40
R_max = 40
m_max = 0
n_max = 0

# === Run and plot ===
Psi_total = compute_total_wavefunction(m_max, n_max, E, a, D, L_max, xi_vals, eta_vals, phi_vals, R_max)
plot_wavefunction_slice(Psi_total, xi_vals, eta_vals, phi_vals)

norm_total = compute_norm(Psi_total, xi_vals, eta_vals, phi_vals, a)
print(f"\nTotal norm of Ψ = {norm_total:.6e}")
