import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv, factorial
from scipy.linalg import eigh
from numpy.polynomial.legendre import leggauss

def solve_angular_equation(m, L_max, E, a, D):
    m = np.abs(m)
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

    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eigh(H, S)

    # Sort by |λ| to define n-ordering correctly
    idx_sorted = np.argsort(np.abs(eigvals))
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    return eigvals, eigvecs, ell_vals


def build_analytic_angular_hamiltonian(m, L_max, E, a, D):
    m = np.abs(m)
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

def plot_comparison(eigvals_num, eigvecs_num, eigvals_an, eigvecs_an, ell_vals, eta, P, num_funcs=4):
    idx_num = np.argsort(np.abs(eigvals_num))
    idx_an  = np.argsort(np.abs(eigvals_an))

    # Adjust figure height depending on number of plots (smaller height per subplot)
    fig, axes = plt.subplots(num_funcs, 1, figsize=(7, 2.0 * num_funcs), sharex=True)

    for n in range(num_funcs):
        v_num = eigvecs_num[:, idx_num[n]]
        v_an  = eigvecs_an[:, idx_an[n]]

        T_num = np.dot(P, v_num)
        T_an  = np.dot(P, v_an)

        # Fix sign ambiguity so value at eta=+1 is positive for each function
        idx_eta_max = np.argmax(eta)
        if T_num[idx_eta_max] < 0:
            T_num = -T_num
        if T_an[idx_eta_max] < 0:
            T_an = -T_an

        ax = axes[n] if num_funcs > 1 else axes
        ax.plot(eta, T_num.real, label=f'Numerical λ={eigvals_num[idx_num[n]]:.3f}', lw=2)
        ax.plot(eta, T_an.real, '--', label=f'Analytic λ={eigvals_an[idx_an[n]]:.3f}', lw=2)
        ax.set_ylabel(r'$T(\eta)$')
        ax.set_ylim(-2, 2)
        ax.grid(True)
        ax.legend()

    plt.xlabel(r'$\eta$')
    plt.suptitle('Angular Eigenfunctions: Numerical vs. Analytic (real part)', y=0.95)
    plt.tight_layout()
    plt.show()


# === Parameters ===
m = 10
L_max = 80
E = 1 / 27.2
a = 2
D = 0.0
eta, w = leggauss(400)

# === Solve ===
eigvals_num, eigvecs_num, ell_vals = solve_angular_equation(m, L_max, E, a, D)
eigvals_an, eigvecs_an, ell_vals_an = build_analytic_angular_hamiltonian(m, L_max, E, a, D)

P = np.array([[lpmv(np.abs(m), l, x) for l in ell_vals] for x in eta])  # basis eval on eta grid

# === Print eigenvalues ===
print("First 6 numerical eigenvalues (sorted by |λ|):")
for i in np.argsort(np.abs(eigvals_num))[:6]:
    print(f"  λ = {eigvals_num[i]:.6f}")

print("\nFirst 6 analytic eigenvalues (sorted by |λ|):")
for i in np.argsort(np.abs(eigvals_an))[:6]:
    print(f"  λ = {eigvals_an[i]:.6f}")

# === Plot comparison ===
plot_comparison(eigvals_num, eigvecs_num, eigvals_an, eigvecs_an, ell_vals, eta, P, num_funcs=4)

def test_spheroidal_orthonormality(eigvecs, ell_vals, m, eta, w, num_modes=6):
    """
    Check orthonormality of Y_{mn} = T_{mn}(eta) * Phi_m(phi) for a fixed m.
    Assumes Phi_m(φ) = exp(i m φ) / sqrt(2π) is already orthonormal in φ.
    """
    print(f"\nTesting orthonormality of Y_{m}n angular modes...")

    # Build Legendre basis on η grid
    P = np.array([[lpmv(m, l, x) for l in ell_vals] for x in eta])  # (N_eta, N_l)

    # Compute T_{mn}(η) = P @ eigvecs
    T_eta_grid = P @ eigvecs  # shape (N_eta, N_modes)

    # Orthonormality integral: ⟨T_{mn}, T_{mn'}⟩ over η
    overlaps = np.zeros((num_modes, num_modes), dtype=np.complex128)
    for i in range(num_modes):
        for j in range(num_modes):
            integrand = np.conj(T_eta_grid[:, i]) * T_eta_grid[:, j]
            overlaps[i, j] = np.sum(w * integrand)  # Gaussian quadrature

    # Multiply by φ integral: ∫₀^{2π} e^{-imφ} e^{imφ} dφ / 2π = 1
    print("Overlap matrix (⟨Y_{mn} | Y_{mn'}⟩):")
    print(np.round(overlaps.real, 6))

    # Check deviation from identity
    identity = np.eye(num_modes)
    err = np.linalg.norm(overlaps - identity)
    print(f"\nDeviation from identity: {err:.2e}")


test_spheroidal_orthonormality(eigvecs_num, ell_vals, m, eta, w, num_modes=10)
