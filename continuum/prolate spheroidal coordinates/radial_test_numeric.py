import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------
# RHS of the Radial Equation
# ---------------------------
def radial_rhs(xi, y, m, c, lam_mn):
    y0, y1 = y
    denom = xi**2 - 1
    rhs0 = y1
    rhs1 = ((lam_mn - c**2 * xi**2 + m**2 / denom) * y0 - 2 * xi * y1) / denom
    return [rhs0, rhs1]

# ---------------------------
# Solve the Radial ODE
# ---------------------------
def solve_radial_ode(m, lam_mn, c, xi_span=(1.0001, 50), num_points=500):
    xi_vals = np.linspace(*xi_span, num_points)
    xi0 = xi_span[0]

    # Improved initial conditions
    if m == 0:
        S0 = 1e-5         # small but nonzero to excite solution
        dS0 = 1e-5        # small slope
    else:
        S0 = (xi0 - 1)**(abs(m) / 2)
        dS0 = (abs(m) / 2) * (xi0 - 1)**(abs(m)/2 - 1)

    sol = solve_ivp(
        lambda xi, y: radial_rhs(xi, y, m, c, lam_mn),
        xi_span, [S0, dS0],
        t_eval=xi_vals, method='RK45',
        rtol=1e-8, atol=1e-10
    )

    return sol.t, sol.y[0]

# ---------------------------
# Plot and Compare
# ---------------------------
if __name__ == "__main__":
    m = 0
    n = 1
    E = 1.0
    a = 1.0
    c = np.sqrt(2 * E * a**2)
    L_max = 40
    D = 0.0

    # Angular eigenvalues
    from compute_angular import build_analytic_angular_hamiltonian
    eigvals_ang, _, _ = build_analytic_angular_hamiltonian(m, L_max, E, a, D)
    lam_mn = np.sort(-(eigvals_ang))[n]


    xi_vals, S_numeric = solve_radial_ode(m, lam_mn, c)

    # Plot normalized
    S_normalized = S_numeric / np.max(np.abs(S_numeric))

    plt.plot(xi_vals, S_normalized, label="Numerical (ODE)", lw=2)
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$S_{m,n}(\xi)$")
    plt.title(rf"Radial Solution for $m={m}, n={n}$ (Numerical ODE)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
