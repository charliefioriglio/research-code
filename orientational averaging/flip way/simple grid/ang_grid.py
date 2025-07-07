import numpy as np

def repulsion_orientations(n_orientations, dg=np.pi / 3, seed=None):
    rng = np.random.default_rng(seed)

    # Estimate number of beta bands
    n_beta = int(np.sqrt(n_orientations))
    db = np.pi / n_beta

    # Add random offset to beta grid (avoiding exact 0 and pi)
    beta_offset = rng.uniform(0, db)
    beta_vals = np.linspace(0, np.pi, n_beta, endpoint=False) + beta_offset
    beta_vals = beta_vals[(beta_vals > 0) & (beta_vals < np.pi)]  # avoid poles

    angle_list = []

    sin_betas = np.sin(beta_vals)
    weights = sin_betas / np.sum(sin_betas)

    for beta, w in zip(beta_vals, weights):
        n_alpha = max(1, int(round(w * n_orientations)))
        alpha_offset = rng.uniform(0, 2 * np.pi)
        alpha_vals = np.linspace(0, 2 * np.pi, n_alpha, endpoint=False) + alpha_offset
        alpha_vals %= 2 * np.pi

        for alpha in alpha_vals:
            # Sample gamma values
            n_gamma = int(round(2 * np.pi / dg))
            gamma_offset = rng.uniform(0, dg)
            gamma_vals = np.linspace(0, 2 * np.pi, n_gamma, endpoint=False) + gamma_offset
            gamma_vals %= 2 * np.pi

            for gamma in gamma_vals:
                angle_list.append([alpha, beta, gamma])

    return np.array(angle_list)
