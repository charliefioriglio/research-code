#HO
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
from scipy.special import hermite
import pandas as pd

# EzDyson Uses atomic units
m_per_au = 5.29177210544e-11
conversion = 1 / np.sqrt(m_per_au)

# Define Constants (SI units)
ka = 1203.22  # Force constant for the anion in N/m
kn = 2401.08  # Force constant for the neutral in N/m
h = 6.626e-34  # Planck's constant in J*s
m = 1.328381e-26  # Mass in kg
hbar = h / (2 * np.pi)  # Reduced Planck's constant
wn = np.sqrt(kn / m)  # Angular frequency
wa = np.sqrt(ka / m)
ra = 1.35e-10 # equilibrium bond length (m)
rn = 1.21e-10

# Define vibrational wavefunction using HO
def V(x, n, w, Req):
    z = np.sqrt(m * w / hbar) * (x - Req) # Unitless position variable
    H = hermite(n)
    N = conversion * (1 / np.sqrt(2**n * math.factorial(n))) * ((m * w) / (np.pi * hbar))**(1 / 4) # au**-1/2 
    WF = np.exp(-z**2 / 2)
    return N * H(z) * WF
    
# Anion
def Va(x):
    return V(x, 0, wa, ra)
    
def Vn(x, n):
    return V(x, n, wn, rn)
    
def total(x, n):
    return (Vn(x, n) * Va(x))**2
    
x_vals = np.linspace(1.24e-10, 1.48e-10, 97)

V0_amps = np.array([total(x, 0) for x in x_vals]).reshape(-1, 1)
V1_amps = np.array([total(x, 1) for x in x_vals]).reshape(-1, 1)
V2_amps = np.array([total(x, 2) for x in x_vals]).reshape(-1, 1)
V3_amps = np.array([total(x, 3) for x in x_vals]).reshape(-1, 1)
V4_amps = np.array([total(x, 4) for x in x_vals]).reshape(-1, 1)
V5_amps = np.array([total(x, 5) for x in x_vals]).reshape(-1, 1)
