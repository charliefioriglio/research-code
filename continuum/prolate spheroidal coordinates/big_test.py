import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, spherical_jn
from compute_single_wavefunction import build_single_mode_wavefunction

def build_spherical_mode(l, m, k, r_vals, theta_vals, phi_vals):
    R_l = spherical_jn(l, k * r_vals)
    Y_lm = sph_harm(m, l, 0, theta_vals)
    Phi_m = np.exp(1j * m * phi_vals) / np.sqrt(2 * np.pi)
    return R_l, Y_lm, Phi_m

def compare_angular_components(theta_vals, eta_vals, phi_vals, T_eta, Φ_prolate, Y_lm, Φ_spherical, m, phi_k=0):

    plt.figure()
    plt.plot(theta_vals, Y_lm, label='|Yₗₘ(θ)| (spherical)')
    eta_from_theta = np.cos(theta_vals)
    plt.plot(theta_vals, (np.interp(eta_from_theta, eta_vals, T_eta)) * 1 / np.sqrt(2 * np.pi), label='|Tₘₙ(η)| (prolate, normalized)')
    plt.title(f"Comparison of Angular Components (m={m})")
    plt.xlabel("θ (rad)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.ylim(-1, 1)
    plt.show()

def compare_radial_components(xi_vals, S_xi, r_vals, R_l, m, l, a):
    # Interpolate S(ξ) onto r_vals using r = a * ξ ⇒ ξ = r / a
    xi_from_r = r_vals / a
    S_interp = np.interp(xi_from_r, xi_vals, S_xi)

    plt.figure()
    plt.plot(r_vals, np.abs(S_interp), label='|Sₘₙ(r/a)| (prolate)')
    plt.plot(r_vals, np.abs(R_l), label=f'|$j_{l}$(kr)| (spherical)')
    plt.xlabel("r")
    plt.title(f"Radial Function Comparison (m={m}, l={l})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def compare_coefficients_prolate_vs_spherical(m, n, l, E, a, D, L_max):
    # Grids
    xi_vals = np.linspace(1.0, 20000.0, 3000)
    eta_vals = np.linspace(-1.0, 1.0, 200)
    phi_vals = np.linspace(0, 2 * np.pi, 120)
    r_vals = np.linspace(1.0, 200.0, 300)
    theta_vals = np.linspace(0, np.pi, 200)

    # Wave parameters
    k = np.sqrt(2 * E)
    theta_k = np.pi / 2
    phi_k = 0.0

    # Build prolate spheroidal wavefunction
    Ψ_mn, S_xi, T_eta, Φ_prolate, coeff, d_coeffs = build_single_mode_wavefunction(
        m, n, E, a, D, L_max, xi_vals, eta_vals, phi_vals, theta_k, phi_k, R_max=40
    )

    # Build spherical counterpart
    R_l, Y_lm, Φ_spherical = build_spherical_mode(l, m, k, r_vals, theta_vals, phi_vals)

    # Angular comparison
    compare_angular_components(theta_vals, eta_vals, phi_vals, T_eta, Φ_prolate, Y_lm, Φ_spherical, m)

    # Radial comparison
    compare_radial_components(xi_vals, S_xi, r_vals, R_l, m, l, a=a)

    # Sample coefficient values at θ=0, φ=0
    eta_k = np.cos(theta_k)
    idx_eta = np.argmin(np.abs(eta_vals - eta_k))
    idx_phi = np.argmin(np.abs(phi_vals - phi_k))

    c_spherical = sph_harm(m, l, phi_k, theta_k) * 4 * np.pi * 1j**l

    print("--- Coefficient Comparison ---")
    print(f"Full projection coeff    : {coeff:.4e}")
    print(f"Spherical Yₗₘ(φ_k,θ_k)   : {c_spherical:.4e}")

# Run comparison
compare_coefficients_prolate_vs_spherical(m=0, n=30, l=30, E=0.1, a=0.01, D=0.0, L_max=40)
