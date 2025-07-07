import numpy as np
from scipy.interpolate import RegularGridInterpolator
from compute_single_wavefunction import build_single_mode_wavefunction
from compute_angular import solve_angular_equation
from scipy.special import lpmv

def compute_continuum_callable(m_max, n_max, E, a, D, L_max,
                                xi_vals, eta_vals, phi_vals,
                                theta_k, phi_k, R_max=50):
    """
    Return a callable ψ(X, Y, Z) giving the spheroidal continuum wavefunction
    in Cartesian space.

    Parameters:
        m_max, n_max: spheroidal mode cutoffs
        E: electron energy (a.u.)
        a: internuclear distance / 2
        D: interaction strength (0 for free electron)
        L_max: max ℓ used in angular basis
        xi_vals, eta_vals, phi_vals: spheroidal coordinate grids
        theta_k, phi_k: direction of emission
        R_max: max radial extent

    Returns:
        ψ_cont(X, Y, Z): callable on 3D Cartesian grids
    """
    xi_grid, eta_grid, phi_grid = np.meshgrid(xi_vals, eta_vals, phi_vals, indexing='ij')
    Psi_total = np.zeros_like(xi_grid, dtype=np.complex128)

    eta_k = np.cos(theta_k)

    for m in range(-m_max, m_max + 1):
        m_abs = abs(m)

        eigvals_ang, eigvecs_ang, ell_vals = solve_angular_equation(m_abs, L_max, E, a, D)
        n_limit = min(n_max, eigvecs_ang.shape[1] - 1)

        for n in range(n_limit + 1):
            Ψ_mn, _, T_eta, Φ_m = build_single_mode_wavefunction(
                m, n, E, a, D, L_max, xi_vals, eta_vals, phi_vals, R_max
            )

            P_basis_k = np.array([lpmv(m_abs, l, eta_k) for l in ell_vals])
            T_eta_k = np.dot(P_basis_k, eigvecs_ang[:, n])
            Φ_m_k = np.exp(1j * m * phi_k) / np.sqrt(2 * np.pi)

            coeff = 4 * np.pi * (1j)**n * T_eta_k * Φ_m_k
            Psi_total += coeff * Ψ_mn

    # Interpolator on prolate spheroidal grid
    interpolator = RegularGridInterpolator(
        (xi_vals, eta_vals, phi_vals), Psi_total, bounds_error=False, fill_value=0.0
    )

    def cartesian_to_prolate_spheroidal(x, y, z, a):
        r1 = np.sqrt(x**2 + y**2 + (z - a)**2)
        r2 = np.sqrt(x**2 + y**2 + (z + a)**2)
        xi = (r1 + r2) / (2 * a)
        eta = (r1 - r2) / (2 * a)
        phi = np.mod(np.arctan2(y, x), 2 * np.pi)
        return xi, eta, phi

    def Ψ_cartesian(X, Y, Z):
        xi, eta, phi = cartesian_to_prolate_spheroidal(X, Y, Z, a)
        pts = np.stack([xi.ravel(), eta.ravel(), phi.ravel()], axis=-1)
        Psi_vals = interpolator(pts).reshape(X.shape)
        return Psi_vals

    return Ψ_cartesian
