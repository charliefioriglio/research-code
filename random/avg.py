import numpy as np
import math
from scipy.integrate import quad
from scipy.special import hermite
import matplotlib.pyplot as plt

# Given constants
ka = 1203.22  # Force constant for the anion in N/m
k = 2401.08  # Force constant for the neutral in N/m
h = 6.626e-34  # Planck's constant in J*s
m = 1.328381e-26  # Mass in kg
hbar = h / (2 * np.pi)  # Reduced Planck's constant
w = np.sqrt(k / m)  # Angular frequency for neutral
wa = np.sqrt(ka / m)  # Angular frequency for anion

# Define the scaled position variables
def xn(x):
    return np.sqrt(m * w / hbar) * (x + 0.14e-10)

def xa(x):
    return np.sqrt(m * wa / hbar) * (x)

# Define the wavefunctions Vn using Hermite polynomials
def Vn(x, n):
    Hn = hermite(n)  # Hermite polynomial of degree n
    return (1 / np.sqrt(2**n * math.factorial(n))) * ((m * w) / (np.pi * hbar))**(1 / 4) * np.exp(-xn(x)**2 / 2) * Hn(xn(x))

def Va(x):
    return ((m * wa) / (np.pi * hbar))**(1 / 4) * np.exp(-xa(x)**2 / 2)

def N(n):
    integral, _ = quad(lambda x: prod(x, n)**2, -.25e-10, .25e-10)
    return 1/((integral))

def prod(x, n):
    return Vn(x, n) * Va(x)

def avg_x(n):
    # Perform the integration over a larger range for better accuracy
    integral, _ = quad(lambda x: x * N(n) * (Vn(x, n) * Va(x))**2, -.25e-10, .25e-10)
    return integral

for n in range(6):
    print(f"⟨x⟩ for n={n}: {avg_x(n)}")

# Calculate FCF's
def fcf(n):
    integral, _ = quad(lambda x: prod(x, n), -.25e-10, .25e-10)
    return integral

for n in range(6):
    print(f"fcf for n={n}: {fcf(n)}")

# Plot Vn * Va for states 0 to 5
x_values = np.linspace(-.25e-10, .25e-10, 500)
for n in range(6):
    y_values = [prod(x, n) for x in x_values]
    plt.plot(x_values * 1e10, y_values, label=f'n={n}')

plt.xlabel('x (Angstroms)')
plt.ylabel('Vn(x) * Va(x)')
plt.title('Product of Vn and Va for Different Quantum States')
plt.legend()
plt.grid(True)
plt.show()

# Derivative
for n in range(6):
    product = prod(x_values, n)
    derivative = np.gradient(product, x_values)
    plt.plot(x_values, derivative)
plt.show()