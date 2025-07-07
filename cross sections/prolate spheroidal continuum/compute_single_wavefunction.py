import numpy as np
from compute_angular import build_analytic_angular_hamiltonian
from compute_radial import compute_radial_function
from compute_phi import compute_Phi_m
from scipy.special import lpmv, factorial

def build_single_mode_wavefunction_on_xyz_grid(m, n, E, a, D, X, Y, Z, L, L_max=50.0, R_max=50.0, xi_vals=None, eta_vals=None, phi_vals=None):

    # Convert Cartesian (X,Y,Z) → prolate spheroidal (ξ, η, φ)
    Z_A = a
    Z_B = -a
    rA = np.sqrt((X)**2 + (Y)**2 + (Z - Z_A)**2)
    rB = np.sqrt((X)**2 + (Y)**2 + (Z - Z_B)**2)

    xi = (rA + rB) / (2 * a)
    eta = (rA - rB) / (2 * a)
    phi = np.arctan2(Y, X)

    # Precompute 1D coordinate grids if not provided
    xi_max = (L / a) * 1.1
    if xi_vals is None:
        xi_vals = np.linspace(1.0, xi_max, 200)
    if eta_vals is None:
        eta_vals = np.linspace(-1.0, 1.0, 200)
    if phi_vals is None:
        phi_vals = np.linspace(-np.pi, np.pi, 200)

    # Solve angular and radial equations
    m_abs = abs(m)
    c = np.sqrt(2 * E * a**2)

    eigvals_ang, eigvecs_ang, ell_vals = build_analytic_angular_hamiltonian(m_abs, L_max, E, a, D)
    v_ang = eigvecs_ang[:, n]

    # Angular function T_{m,n}(η)
    P_basis = np.array([lpmv(m_abs, l, eta_vals) for l in ell_vals])
    T_eta_vals = P_basis.T @ v_ang
    if T_eta_vals[-1].real < 0:
        v_ang *= -1
        T_eta_vals = P_basis.T @ v_ang
    if m < 0:
        T_eta_vals = T_eta_vals * (-1)**m_abs

    # Interpolate T_eta onto grid η
    from scipy.interpolate import interp1d
    T_eta_func = interp1d(eta_vals, T_eta_vals, kind='cubic', fill_value=0, bounds_error=False)
    T_eta = T_eta_func(eta)

    # Radial function S_{m,n}(ξ)
    _, S_xi_vals, _, _, _ = compute_radial_function(m, n, eigvals_ang, c, R_max, xi_vals)
    S_xi_func = interp1d(xi_vals, S_xi_vals, kind='cubic', fill_value=0, bounds_error=False)
    S_xi = S_xi_func(xi)

    # Azimuthal part
    Φ_m = compute_Phi_m(m, phi)

    # Final wavefunction on (X, Y, Z)
    Ψ_mn = S_xi * T_eta * Φ_m

    return Ψ_mn
