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
            Ψ_mn, _, _, _ = build_single_mode_wavefunction(
                m, n, E, a, D, L_max, xi_vals, eta_vals, phi_vals, theta_k, phi_k, R_max
            )
         
            # Add contribution
            Psi_total += Ψ_mn

    return Psi_total

def plot_total_radial_angular_parts(m_max, n_max, E, a, D, L_max, xi_vals, eta_vals, phi_vals, theta_k, phi_k, R_max=50):
    from compute_angular import solve_angular_equation
    from compute_radial import compute_radial_function
    from compute_phi import compute_Phi_m
    from scipy.special import lpmv
    from matplotlib import cm

    c = np.sqrt(2 * E * a**2)
    eta_k = np.cos(theta_k)

    radial_sum = np.zeros_like(xi_vals, dtype=complex)
    angular_grid = np.zeros((len(eta_vals), len(phi_vals)), dtype=complex)

    for m in range(-m_max, m_max + 1):
        m_abs = abs(m)

        eigvals_ang, eigvecs_ang, ell_vals = solve_angular_equation(m, L_max, E, a, D)
        n_limit = min(n_max, eigvecs_ang.shape[1] - 1)

        Φ_m_vals = np.exp(1j * m * phi_vals) / np.sqrt(2 * np.pi)

        for n in range(0, n_limit + 1):
            # === Radial Part ===
            xi, S_mn_xi, _, _, _ = compute_radial_function(m, n, eigvals_ang, c, R_max, xi_vals)

            # === Angular Part ===
            P_basis_k = np.array([lpmv(m_abs, l, eta_k) for l in ell_vals])
            T_eta_k = np.dot(P_basis_k, eigvecs_ang[:, n])

            P_eta = np.array([[lpmv(m_abs, l, eta) for l in ell_vals] for eta in eta_vals])
            T_eta_vals = P_eta @ eigvecs_ang[:, n]

            # Adjust for negative m (parity)
            if m < 0:
                T_eta_vals *= (-1)**m_abs

            coeff = 4 * np.pi * (1j)**n * T_eta_k * np.exp(1j * m * phi_k) / np.sqrt(2 * np.pi)

            # Accumulate radial and angular contributions separately
            radial_sum += coeff * np.trapz(T_eta_vals * np.trapz(Φ_m_vals, phi_vals), eta_vals) * S_mn_xi

            for i_eta, eta in enumerate(eta_vals):
                angular_grid[i_eta, :] += coeff * T_eta_vals[i_eta] * Φ_m_vals

    # === Plot Radial Part ===
    plt.figure(figsize=(6, 4))
    plt.plot(xi_vals, radial_sum.real, label="Re", lw=2)
    plt.plot(xi_vals, radial_sum.imag, label="Im", lw=2)
    plt.title("Total Radial Component")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"Radial Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Plot Angular Part ===
    ETA, PHI = np.meshgrid(eta_vals, phi_vals, indexing='ij')
    plt.figure(figsize=(7, 5))
    plt.pcolormesh(PHI, ETA, angular_grid.real, shading='auto', cmap=cm.RdBu)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\eta$")
    plt.title("Re[Total Angular Component]")
    plt.colorbar(label="Re[$\Psi_{\text{angular}}$]")
    plt.tight_layout()
    plt.show()


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
xi_vals = np.linspace(1.0, 100.0, 300)
eta_vals = np.linspace(-1.0, 1.0, 200)
phi_vals = np.linspace(0, 2 * np.pi, 120)

E = 0.1
a = 1
D = 0.0
L_max = 60
R_max = 60
m_max = 0
n_max = 0

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
                                 x_fixed=0.0, ylim=(-20, 20), zlim=(-20, 20), N=200)

plot_total_radial_angular_parts(
    m_max,
    n_max,
    E,
    a,
    D,
    L_max,
    xi_vals,
    eta_vals,
    phi_vals,
    theta_k,
    phi_k
)
