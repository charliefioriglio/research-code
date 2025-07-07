
#morse_beta
import numpy as np
import pandas as pd
from morse import V0_amps, V1_amps, V2_amps, V3_amps, V4_amps, V5_amps
from par import par0, par1, par2, par3, par4, par5
from perp import perp0, perp1, perp2, perp3, perp4, perp5


sigma_par_V0 = (par0 @ V0_amps)
sigma_par_V1 = (par1 @ V1_amps)
sigma_par_V2 = (par2 @ V2_amps)
sigma_par_V3 = (par3 @ V3_amps)
sigma_par_V4 = (par4 @ V4_amps)
sigma_par_V5 = (par5 @ V5_amps)

sigma_perp_V0 = (perp0 @ V0_amps)
sigma_perp_V1 = (perp1 @ V1_amps)
sigma_perp_V2 = (perp2 @ V2_amps)
sigma_perp_V3 = (perp3 @ V3_amps)
sigma_perp_V4 = (perp4 @ V4_amps)
sigma_perp_V5 = (perp5 @ V5_amps)


numerator_V0 = 2 * (sigma_par_V0 - sigma_perp_V0)
denominator_V0 = sigma_par_V0 + 2 * sigma_perp_V0
numerator_V1 = 2 * (sigma_par_V1 - sigma_perp_V1)
denominator_V1 = sigma_par_V1 + 2 * sigma_perp_V1
numerator_V2 = 2 * (sigma_par_V2 - sigma_perp_V2)
denominator_V2 = sigma_par_V2 + 2 * sigma_perp_V2
numerator_V3 = 2 * (sigma_par_V3 - sigma_perp_V3)
denominator_V3 = sigma_par_V3 + 2 * sigma_perp_V3
numerator_V4 = 2 * (sigma_par_V4 - sigma_perp_V4)
denominator_V4 = sigma_par_V4 + 2 * sigma_perp_V4
numerator_V5 = 2 * (sigma_par_V5 - sigma_perp_V5)
denominator_V5 = sigma_par_V5 + 2 * sigma_perp_V5

beta_V0 = numerator_V0 / denominator_V0
beta_V1 = numerator_V1 / denominator_V1
beta_V2 = numerator_V2 / denominator_V2
beta_V3 = numerator_V3 / denominator_V3
beta_V4 = numerator_V4 / denominator_V4
beta_V5 = numerator_V5 / denominator_V5

matrix = np.hstack((beta_V0, beta_V1, beta_V2, beta_V3, beta_V4, beta_V5))

# Convert each beta matrix to a DataFrame and export as CSV
pd.DataFrame(matrix).to_csv('morse_beta.csv', index=False)

