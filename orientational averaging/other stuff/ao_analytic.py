import numpy as np
from scipy.special import spherical_jn, genlaguerre, factorial
from scipy.integrate import quad
import matplotlib.pyplot as plt

# H atom radial function
def R_nl_hydrogenic(r, n, l, Z=1):
    rho = 2 * Z * r / n
    prefactor = np.sqrt((2 * Z / n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)
    return prefactor * rho**l * np.exp(-rho / 2) * laguerre_poly(rho)

# Compute ∫₀^∞ j_{l±1}(kr) * r * R_{n,l}(r) dr
def sigma(n, l, k, delta_l, Z=1):
    l2 = l + delta_l
    if l2 < 0:
        raise ValueError("l + delta_l must be ≥ 0.")
    
    integrand = lambda r: spherical_jn(l2, k * r) * r * R_nl_hydrogenic(r, n, l, Z)
    result, _ = quad(integrand, 0, np.inf, limit=1000)
    return result

# beta = l * (l - 1) * sigma(l-1)**2 + (l + 1) * (l + 2) * sigma(l+1)**2 - 6 * l * (l + 1) * sigma(l-1) * sigma(l+1) / (2 * l + 1) * (l * sigma(l-1)**2 + (l + 1) * sigma(l+1)**2)

# Calculate
# Constants
hartree = 27.2114  # 1 Hartree in eV

# Quantum numbers and effective Z
n = 3
l = 2
Z_eff = 1

# Energy grid in eV
E_eV = np.linspace(0.01, 2.5, 20)
beta_vals = []

# Compute β for each energy
for E in E_eV:
    k = np.sqrt(2 * E / hartree)

    sig_l_minus_1 = sigma(n, l, k, delta_l=-1, Z=Z_eff)
    sig_l_plus_1 = sigma(n, l, k, delta_l=+1, Z=Z_eff)

    sigm1_sq = sig_l_minus_1**2
    sigp1_sq = sig_l_plus_1**2
    sigm1p1 = sig_l_minus_1 * sig_l_plus_1

    numerator = (
        l * (l - 1) * sigm1_sq +
        (l + 1) * (l + 2) * sigp1_sq -
        6 * l * (l + 1) * sigm1p1
    )
    denominator = (2 * l + 1) * (l * sigm1_sq + (l + 1) * sigp1_sq)

    beta = numerator / denominator
    beta_vals.append(beta)

# Plot result
plt.figure(figsize=(8, 5))
plt.plot(E_eV, beta_vals, 'o-')
plt.xlabel('Kinetic Energy (eV)')
plt.ylabel(r'$\beta$')
plt.ylim(-1, 2)
plt.title(r'Anisotropy Parameter $\beta$ vs Electron Kinetic Energy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()