import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Define Values
h_bar = 1
m = 1
k = 1
s = 0.01  # step size

x = np.arange(-6, 6.01, s)

H = np.zeros((len(x), len(x)))  # Initialize an n x n matrix for the Hamiltonian

V = 1/2 * k * x**2  # Define Potential

# Set up Potential matrix | V(xn) along diagonal of H
#np.fill_diagonal(H, V)

for i in range(len(x)):
    for j in range(len(x)):
        if i==j:
            H[i, j] = V[i]

# Set up Kinetic Matrix
T = (-h_bar**2 / (2 * m * s**2)) * (-2 * np.diag(np.ones(len(x))) + np.diag(np.ones(len(x)-1), 1) + np.diag(np.ones(len(x)-1), -1))

# Combine H = T + V
H = H + T

E, Psi = la.eigh(H)

print(E[0], E[1], E[2])

plt.plot(x, V, "black")
plt.axhline(E[0])
plt.plot(x, Psi[:, 0] + E[0], label="v = 0")
plt.axhline(E[1])
plt.plot(x, Psi[:, 1] + E[1], label="v = 1")
plt.axhline(E[2])
plt.plot(x, Psi[:, 2] + E[2], label="v = 1")
plt.xlim(-4, 4)
plt.ylim(0, 4)
plt.legend()
plt.show()