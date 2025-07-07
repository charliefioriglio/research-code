#HO

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Given constants
ka = 1203.22  # Force constant for the anion in N/m
k = 2401.08  # Force constant for the neutral in N/m
h = 6.626e-34  # Planck's constant in J*s
m = 1.328381e-26  # Mass in kg
hbar = h / (2 * np.pi)  # Reduced Planck's constant
w = np.sqrt(k / m)  # Angular frequency
wa = np.sqrt(ka / m)
# Define x symbolically
x = sp.symbols('x')

# Define the scaled position variable
xi = sp.sqrt(m * w / hbar) * (x - 1.22e-10)
xa = sp.sqrt(m * wa / hbar) * (x - 1.34e-10)

# Define the Hermite polynomials for each wavefunction
H0 = sp.hermite(0, xi)
H1 = sp.hermite(1, xi)
H2 = sp.hermite(2, xi)
H3 = sp.hermite(3, xi)
H4 = sp.hermite(4, xi)
H5 = sp.hermite(5, xi)
Ha = sp.hermite(0, xa)

# Define the wavefunctions Vn
V0 = (1 / sp.sqrt(2**0 * sp.factorial(0))) * ((m * w)/(np.pi * hbar))**(1 / 4) * sp.exp(-xi**2 / 2) * H0 * 1/10
V1 = (1 / sp.sqrt(2**1 * sp.factorial(1))) * ((m * w)/(np.pi * hbar))**(1 / 4) * sp.exp(-xi**2 / 2) * H1 * 1/10
V2 = (1 / sp.sqrt(2**2 * sp.factorial(2))) * ((m * w)/(np.pi * hbar))**(1 / 4) * sp.exp(-xi**2 / 2) * H2  * 1/10
V3 = (1 / sp.sqrt(2**3 * sp.factorial(3))) * ((m * w)/(np.pi * hbar))**(1 / 4) * sp.exp(-xi**2 / 2) * H3 * 1/10
V4 = (1 / sp.sqrt(2**4 * sp.factorial(4))) * ((m* w)/(np.pi * hbar))**(1 / 4) * sp.exp(-xi**2 / 2) * H4 * 1/10
V5 = (1 / sp.sqrt(2**5 * sp.factorial(5))) * ((m * w)/(np.pi * hbar))**(1 / 4) * sp.exp(-xi**2 / 2) * H5 * 1/10
Va = (1 / sp.sqrt(2**0 * sp.factorial(0))) * ((m * wa)/(np.pi * hbar))**(1 / 4) * sp.exp(-xa**2 / 2) * Ha * 1/10

f0 = sp.lambdify(x, V0, 'numpy')
f1 = sp.lambdify(x, V1, 'numpy')
f2 = sp.lambdify(x, V2, 'numpy')
f3 = sp.lambdify(x, V3, 'numpy')
f4 = sp.lambdify(x, V4, 'numpy')
f5 = sp.lambdify(x, V5, 'numpy')
fa = sp.lambdify(x, Va, 'numpy')

x_values = np.linspace(1.24e-10, 1.48e-10, 1000)

y0 = f0(x_values)
y1 = f1(x_values)
y2 = f2(x_values)
y3 = f3(x_values)
y4 = f4(x_values)
y5 = f5(x_values)
ya = fa(x_values)

m0 = sp.Matrix(y0).reshape(len(y0), 1)
m1 = sp.Matrix(y1).reshape(len(y1), 1)
m2 = sp.Matrix(y2).reshape(len(y2), 1)
m3 = sp.Matrix(y3).reshape(len(y3), 1)
m4 = sp.Matrix(y4).reshape(len(y4), 1)
m5 = sp.Matrix(y5).reshape(len(y5), 1)
ma = sp.Matrix(ya).reshape(len(ya), 1)

V0_a = (m0.multiply_elementwise(ma))
V1_a = (m1.multiply_elementwise(ma))
V2_a = (m2.multiply_elementwise(ma))
V3_a = (m3.multiply_elementwise(ma))
V4_a = (m4.multiply_elementwise(ma))
V5_a = (m5.multiply_elementwise(ma))

V0_amps = (V0_a.multiply_elementwise(V0_a))
V1_amps = (V1_a.multiply_elementwise(V1_a))
V2_amps = (V2_a.multiply_elementwise(V2_a))
V3_amps = (V3_a.multiply_elementwise(V3_a))
V4_amps = (V4_a.multiply_elementwise(V4_a))
V5_amps = (V5_a.multiply_elementwise(V5_a))

# plot
plt.plot(x_values, V0_amps, label="V0")
plt.plot(x_values, V1_amps, label="V1")
plt.plot(x_values, V2_amps, label="V2")
plt.plot(x_values, V3_amps, label="V3")
plt.plot(x_values, V4_amps, label="V4")
plt.plot(x_values, V5_amps, label="V5")
plt.xlabel('r (Å)')
plt.ylabel('overlap (cm^-2)')
plt.title('HO')
plt.legend()
plt.axhline(0, color='black')
plt.show()


#morse
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Insert D, v, and r values
Da = 6.55e-19 #J
ka = 1203.22 #N/m
ra = 1.35e-10 #m

Dn = 8.20e-19
kn = 2401.08
rn = 1.21e-10

h = 6.626e-34 #Js
m = 1.328381e-26 #kg
hbar = h / (2 * np.pi)

unit_conversion = 1/10 #1/m^.5 to 1/cm^.5
# Define parameters
aa = np.sqrt(ka / (2 * Da))
la = np.sqrt(2 * m * Da) / (aa * hbar)

an = np.sqrt(kn / (2 * Dn))
ln = np.sqrt(2 * m * Dn) / (an * hbar)

# Define x and n
x = sp.symbols('x')

# Define r_values range with 1000 points
r_values = np.linspace(1.24, 1.49, 1000)

# Define Anion Wavefunction
na = 0
ia = 2 * la - 2 * na - 1
La = sp.assoc_laguerre(na, ia, 2 * la * sp.exp(-aa * x))
Na = unit_conversion * sp.sqrt((sp.factorial(na) * (2 * la - 2 * na - 1) * aa) / (sp.gamma(2 * la - na)))
Va = Na * (2 * la * sp.exp(-aa * x))**(la - na - 0.5) * sp.exp(-0.5 * (2 * la * sp.exp(-aa * x))) * La
Va_values = []

# Evaluate Va at each r_value
for r_value in r_values:
    xa = (r_value * 1e-10) - ra
    Va_value = Va.subs(x, xa)    
    Va_values.append(Va_value)
Va_values = np.array(Va_values).reshape(1000, 1)
Va_amps = Va_values**2

# Initialize matrices dictionary
V_matrices = {}

# Loop over nn values from 0 to 5
for nn in range(6):
    # Define Neutral Wavefunction
    iN = 2 * ln - 2 * nn - 1
    Ln = sp.assoc_laguerre(nn, iN, 2 * ln * sp.exp(-an * x))
    Nn = unit_conversion * sp.sqrt((sp.factorial(nn) * (2 * ln - 2 * nn - 1) * an) / (sp.gamma(2 * ln - nn)))
    Vn = Nn * (2 * ln * sp.exp(-an * x))**(ln - nn - 0.5) * sp.exp(-0.5 * (2 * ln * sp.exp(-an * x))) * Ln

    # Initialize result list
    Vn_values = []

    # Evaluate V at each r_value
    for r_value in r_values:
        xn = (r_value * 1e-10) - rn
        Vn_value = Vn.subs(x, xn)
        Vn_values.append(Vn_value)
    Vn_matrix = np.array(Vn_values).reshape(1000, 1)
    
    # Store the matrix with the name V0, V1, V2, etc.
    V_matrices[f'V{nn}'] = Vn_matrix

# Access the matrices as V0, V1, V2, V3, V4, V5
V0 = V_matrices['V0']
V1 = V_matrices['V1']
V2 = V_matrices['V2']
V3 = V_matrices['V3']
V4 = V_matrices['V4']
V5 = V_matrices['V5']

# Element-wise multiplication for amps
V0_amps = (V0 * Va_values)**2
V1_amps = (V1 * Va_values)**2
V2_amps = (V2 * Va_values)**2
V3_amps = (V3 * Va_values)**2
V4_amps = (V4 * Va_values)**2
V5_amps = (V5 * Va_values)**2

# Plotting Vx_amps vs r_values for x = 0, 1, 2, 3, 4, 5

plt.plot(r_values, V0_amps, label='V0_amps')
plt.plot(r_values, V1_amps, label='V1_amps')
plt.plot(r_values, V2_amps, label='V2_amps')
plt.plot(r_values, V3_amps, label='V3_amps')
plt.plot(r_values, V4_amps, label='V4_amps')
plt.plot(r_values, V5_amps, label='V5_amps')

plt.xlabel('r (Å)')
plt.ylabel('overlap')
plt.title('morse')
plt.legend()
plt.axhline(0, color='black')
plt.show()


