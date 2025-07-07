import numpy as np
import matplotlib.pyplot as plt
from compute_wavefunction import compute_continuum_callable

# -----------------------------------------------
# Parameters
# -----------------------------------------------
a = 0.675 / 52.9e-2
E = 3 / 27.2
D = 0.0
m_max = 10
n_max = 10
L_max = 50
R_max = 50

# Cartesian grid
L = 20.0
n_pts = 200
x = np.linspace(-L, L, n_pts)
y = np.linspace(-L, L, n_pts)
z = np.linspace(-L, L, n_pts)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Evaluate on YZ slice at x = 0
x0_index = np.argmin(np.abs(x))  # Find index where x ≈ 0
x0 = x[x0_index]
Y_slice, Z_slice = np.meshgrid(y, z, indexing='ij')
YZ_points = np.stack([np.full_like(Y_slice, x0), Y_slice, Z_slice], axis=-1)

# -----------------------------------------------
# Define incoming directions
# -----------------------------------------------
angles = [
    (0.0, 0.0),
    (np.pi / 2, 0.0),
    (np.pi / 2, np.pi / 2)
]

labels = [
    r"$\theta_k=0$, $\phi_k=0$",
    r"$\theta_k=\pi/2$, $\phi_k=0$",
    r"$\theta_k=\pi/2$, $\phi_k=\pi/2$"
]

# -----------------------------------------------
# Plotting
# -----------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for ax, (theta_k, phi_k), label in zip(axs, angles, labels):
    # Compute interpolator
    wavefunction_interp = compute_continuum_callable(
        m_max=m_max, n_max=n_max, E=E, a=a, D=D, L_max=L_max,
        X=X, Y=Y, Z=Z, theta_k=theta_k, phi_k=phi_k, R_max=R_max
    )

    # Evaluate on slice
    Psi_slice = wavefunction_interp(YZ_points)
    amp_squared = np.abs(Psi_slice.T)**2

    im = ax.imshow(amp_squared, origin='lower', extent=[y[0], y[-1], z[0], z[-1]],
                   cmap='viridis', aspect='auto')
    ax.set_title(f"|Ψ(x=0, y, z)|²\n{label}")
    ax.set_xlabel("y (a.u.)")
    ax.set_ylabel("z (a.u.)")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
