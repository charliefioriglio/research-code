
#morse
import numpy as np
import pandas as pd
import math
from scipy.special import eval_genlaguerre
from scipy.integrate import quad
from scipy.special import gamma

# EzDyson Uses atomic units
m_per_au = 5.29177210544e-11
conversion = 1 / np.sqrt(m_per_au)

# Constants (SI units)
h = 6.626e-34
c = 2.998e8
m = 1.328381e-26
hbar = h / (2 * np.pi)

# Anion
Da = 6.55e-19 # Disociation Energy
ka = 1203.22 # Force Constant
ra = 1.35e-10 # Equilibrium Bond Length

# Neutral
Dn = 8.20e-19 # J
kn = 2401.08 # J * m**-2
rn = 1.21e-10 # m

# Define parameters to allow arguments to be unitless
aa = np.sqrt(ka/(2*Da)) # m**-1
la = np.sqrt(2 * m * Da) / (aa * hbar) # Unitless

an = np.sqrt(kn/(2*Dn))
ln = np.sqrt(2 * m * Dn) / (an * hbar)

# Define Morse Wavefunction
def V(x, n, a, l, Req):
    z = 2 * l * np.exp(-a * (x - Req))
    i = 2 * l - 2 * n - 1 # upper index for the associated laguerre polynomial
    L = eval_genlaguerre(n, i, z)
    N = np.sqrt((math.factorial(n) * (2 * l - 2 * n - 1) * a) / (gamma(2 * l - n))) * conversion # Normalization Coefficient (au**-1/2)
    WF = (z**(l - n - 1/2) * np.exp(-1/2 * z)) # Unitless
    return N * WF * L # au**-1/2

# Anion
def Va(x):
    return V(x, 0, aa, la, ra) # au**-1/2

# Neutral
def Vn(x, n):
    return V(x, n, an, ln, rn) # au**-1/2

# Take the sqare of the product of the amplitudes
def total(x, n):
    return (Vn(x, n) * Va(x))**2 # au**-2

r_vals = np.linspace(1.24e-10, 1.48e-10, 97)

V0_amps = np.array([total(x, 0) for x in r_vals]).reshape(-1, 1)
V1_amps = np.array([total(x, 1) for x in r_vals]).reshape(-1, 1)
V2_amps = np.array([total(x, 2) for x in r_vals]).reshape(-1, 1)
V3_amps = np.array([total(x, 3) for x in r_vals]).reshape(-1, 1)
V4_amps = np.array([total(x, 4) for x in r_vals]).reshape(-1, 1)
V5_amps = np.array([total(x, 5) for x in r_vals]).reshape(-1, 1)
