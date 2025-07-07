import numpy as np
from compute_angular import build_analytic_angular_hamiltonian
from compute_radial import compute_radial_function
from compute_phi import compute_Phi_m
from scipy.special import lpmv, factorial

def build_single_mode_wavefunction(m, n, E, a, D, L_max, xi_vals, eta_vals, phi_vals, theta_k, phi_k, R_max):

    m_abs = abs(m)
    c = np.sqrt(2 * E * a**2)
    xi_grid, eta_grid, phi_grid = np.meshgrid(xi_vals, eta_vals, phi_vals, indexing='ij')

    eigvals_ang, eigvecs_ang, ell_vals = build_analytic_angular_hamiltonian(m_abs, L_max, E, a, D)
    v_ang = eigvecs_ang[:, n]

    # Angular function T_{m,n}(η)
    P_basis = np.array([lpmv(m_abs, l, eta_vals) for l in ell_vals])  # shape (N_basis, len(eta))

    T_eta = P_basis.T @ v_ang  # shape (len(eta),)

    # Fix sign: ensure T_eta(eta=1) > 0
    if T_eta[-1].real < 0:  # eta=1 is at the last index since eta_vals spans [-1,1]
        v_ang *= -1
        T_eta = P_basis.T @ v_ang

    if m < 0:
        T_eta = T_eta * (-1)**m_abs


    # Radial function S_{m,n}(ξ)
    _, S_xi, d_coeffs, _, _ = compute_radial_function(m, n, eigvals_ang, c, R_max, xi_vals)

    # Azimuthal function Φ_m(φ)
    Φ_m = compute_Phi_m(m, phi_vals)  # Use signed m here for phase correctness

    # Outer product to build full Ψ
    Ψ_mn_no_coeff = np.outer(S_xi, T_eta)[:, :, np.newaxis] * Φ_m[np.newaxis, np.newaxis, :]

    # Calculate Coefficient
    eta_k = np.cos(theta_k)
    P_basis_k = np.array([lpmv(m_abs, l, eta_k) for l in ell_vals])
    T_eta_k = np.dot(P_basis_k, eigvecs_ang[:, n])  # Scalar
    if m < 0:
       T_eta_k = T_eta_k * (-1)**m_abs

    # Evaluate Φ_m(φ_k)
    Φ_m_k = np.exp(1j * m * phi_k) / np.sqrt(2 * np.pi)

    # Full coefficient
    coeff = 4 * np.pi * 1j**(n + m_abs) * T_eta_k * Φ_m_k

    Ψ_mn = coeff * Ψ_mn_no_coeff

    return Ψ_mn, S_xi, T_eta, Φ_m, coeff, d_coeffs
