import numpy as np
from scipy.special import lpmv, factorial
from scipy.linalg import eigh
from numpy.polynomial.legendre import leggauss

def solve_angular_equation(m, L_max, E, a, D):
    m = abs(m)
    c = np.sqrt(2 * E * a**2)
    ell_vals = np.arange(m, L_max + 1)
    N = len(ell_vals)

    # Overlap matrix S (analytic diagonal)
    S_diag = np.array([2 * factorial(l + m) / ((2 * l + 1) * factorial(l - m)) for l in ell_vals])
    S = np.diag(S_diag)

    # Eta operator matrix in P_l^m basis from recursion relation
    eta_mat = np.zeros((N, N))
    for i, l in enumerate(ell_vals):
        denom = 2 * l + 1
        if i + 1 < N:
            A = (l + 1 - m) / denom
            eta_mat[i + 1, i] = A
            eta_mat[i, i + 1] = A
        if i - 1 >= 0:
            B = (l + m) / denom
            eta_mat[i - 1, i] = B
            eta_mat[i, i - 1] = B

    # Kinetic energy: diagonal matrix -l(l+1) * S_diag
    L2_diag = -ell_vals * (ell_vals + 1) * S_diag
    T = np.diag(L2_diag)

    # Potential operator: -c^2 eta^2 - 2 D eta
    eta2_mat = eta_mat @ eta_mat
    V = -c**2 * eta2_mat - 2 * D * eta_mat

    # Full Hamiltonian
    H = T + V

    # Solve generalized eigenvalue problem H v = λ S v
    eigvals, eigvecs = eigh(H, S)

    # Sort by |λ| to define n-ordering correctly
    idx_sorted = np.argsort(np.abs(eigvals))
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    return eigvals, eigvecs, ell_vals

def build_analytic_angular_hamiltonian(m, L_max, E, a, D):
    m = abs(m)
    c = np.sqrt(2 * E * a**2)
    ell_vals = np.arange(m, L_max + 1)
    N = len(ell_vals)
    H = np.zeros((N, N))
    S_diag = np.array([2 * factorial(l + m) / ((2 * l + 1) * factorial(l - m)) for l in ell_vals])
    S = np.diag(S_diag)

    for i, l in enumerate(ell_vals):
        if l - m >= 0:
            f1 = (l * (l + 1)) * S_diag[i]
            f2 = c**2 * ((l + m) * (l - m) / ((2 * l + 1) * (2 * l - 1)) +
                         (l - m + 1) * (l + m + 1) / ((2 * l + 1) * (2 * l + 3)))
            f2 *= 2 * factorial(l + m) / ((2 * l + 1) * factorial(l - m))
            H[i, i] += -f1 - f2
        if i + 1 < N:
            f = (-2 * D / (2 * l + 1)) * (l + 1 - m)
            f *= 2 * factorial(l + m + 1) / ((2 * l + 3) * factorial(l - m + 1))
            H[i + 1, i] += f
            H[i, i + 1] += f
        if i - 1 >= 0:
            f = (-2 * D / (2 * l + 1)) * (l + m)
            f *= 2 * factorial(l + m - 1) / ((2 * l - 1) * factorial(l - m - 1))
            H[i - 1, i] += f
            H[i, i - 1] += f
        if i + 2 < N:
            f = -c**2 * (l - m + 1) * (l - m + 2)
            f *= 2 * factorial(l + m + 2) / (
                (2 * l + 1) * (2 * l + 3) * (2 * l + 5) * factorial(l - m + 2)
            )
            H[i + 2, i] += f
            H[i, i + 2] += f
        if i - 2 >= 0:
            f = -c**2 * (l + m) * (l + m - 1)
            f *= 2 * factorial(l + m - 2) / (
                (2 * l + 1) * (2 * l - 1) * (2 * l - 3) * factorial(l - m - 2)
            )
            H[i - 2, i] += f
            H[i, i - 2] += f

    eigvals, eigvecs = eigh(H, S)

    # Sort by |λ| to define n-ordering correctly
    idx_sorted = np.argsort(np.abs(eigvals))
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    return eigvals, eigvecs, ell_vals