import numpy as np
from scipy.interpolate import RegularGridInterpolator
from compute_single_wavefunction import build_single_mode_wavefunction_on_xyz_grid
from compute_angular import solve_angular_equation
from scipy.special import lpmv

def compute_continuum_callable(m_max, n_max, E, a, D, L_max,
                                X, Y, Z,
                                theta_k, phi_k, R_max=50.0):
    
    Psi_total = np.zeros_like(X, dtype=np.complex128)

    for m in range(-m_max, m_max + 1):
        m_abs = abs(m)

        # Solve angular Hamiltonian
        eigvals_ang, eigvecs_ang, ell_vals = solve_angular_equation(m_abs, L_max, E, a, D)
        n_limit = min(n_max, eigvecs_ang.shape[1] - 1)

        for n in range(n_limit + 1):
            # Get single (m,n) wavefunction on the Cartesian grid
            Ψ_mn = build_single_mode_wavefunction_on_xyz_grid(
                m, n, E, a, D, X, Y, Z, L=np.max(np.abs([X, Y, Z])),  # Use extent of grid
                theta_k=theta_k, phi_k=phi_k,
                L_max=L_max, R_max=R_max
            )

            Psi_total += Ψ_mn

    # Build interpolator
    x_vals = X[:, 0, 0]
    y_vals = Y[0, :, 0]
    z_vals = Z[0, 0, :]

    interpolator = RegularGridInterpolator(
        (x_vals, y_vals, z_vals), Psi_total, bounds_error=False, fill_value=0.0
    )

    return interpolator
