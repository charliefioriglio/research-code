import numpy as np
from scipy.special import spherical_jn, spherical_yn, factorial
from scipy.linalg import eigh

# ---------------------------
# Build Radial Coefficient Matrix
# ---------------------------
def build_radial_coefficient_matrix(m, lam_mn, c, R_max, parity):
    m_abs = abs(m)

    if parity == 'even':
        r_vals = np.arange(0, 2 * R_max, 2)
    elif parity == 'odd':
        r_vals = np.arange(1, 2 * R_max, 2)
    else:
        raise ValueError("parity must be 'even' or 'odd'")

    N = len(r_vals)
    A = np.zeros((N, N))

    for i, r in enumerate(r_vals):
        l = m_abs + r

        # Diagonal element
        A[i, i] = l * (l + 1) - lam_mn
        if r >= 1:
            A[i, i] += ((2 * l * (l + 1) - 2 * m_abs**2 - 1) * c**2) / ((2 * l - 1) * (2 * l + 3))

        # Upper diagonal (r → r+2)
        if i + 1 < N:
            A[i, i + 1] = ((2 * m_abs + r + 2) * (2 * m_abs + r + 1) * c**2) / (
                (2 * m_abs + 2 * r + 3) * (2 * m_abs + 2 * r + 5)
            )
            A[i + 1, i] = A[i, i + 1]  # Symmetric

        # Lower diagonal (r → r−2)
        if i - 1 >= 0:
            A[i, i - 1] = (r * (r - 1) * c**2) / (
                (2 * m_abs + 2 * r - 3) * (2 * m_abs + 2 * r - 1)
            )
            A[i - 1, i] = A[i, i - 1]

    return A, r_vals

# ---------------------------
# Compute Radial Function
# ---------------------------
def compute_radial_function(m, n, eigvals_ang, c, R_max, xi_vals):
    m_abs = abs(m)
    eigvals_ang = np.argsort(np.abs(eigvals_ang))
    lam_mn = -eigvals_ang[n]
    parity = 'even' if (n - m_abs) % 2 == 0 else 'odd'

    A, r_vals = build_radial_coefficient_matrix(m_abs, lam_mn, c, R_max, parity)
    eigvals, eigvecs = eigh(A)

    # Sort eigenvalues and take the nth root
    sorted_indices = np.argsort(eigvals)
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]
    d_coeffs = eigvecs_sorted[:, n]

    # Normalization (Flammer-style)
    norm_sum = np.sum([
        d * factorial(2 * m_abs + r) / factorial(r)
        for d, r in zip(d_coeffs, r_vals)
    ])
    norm = 1.0 / norm_sum

    prefactor = ((xi_vals**2 - 1) / xi_vals**2)**(m_abs / 2)
    sum_series = np.zeros_like(xi_vals, dtype=complex)

    for d, r in zip(d_coeffs, r_vals):
        l = m_abs + r
        coeff = factorial(2 * m_abs + r) / factorial(r)
        j_l = spherical_jn(l, c * xi_vals)
        sum_series += d * coeff * j_l

    S_mn_xi = norm * prefactor * sum_series

    return xi_vals, S_mn_xi, d_coeffs, r_vals, parity