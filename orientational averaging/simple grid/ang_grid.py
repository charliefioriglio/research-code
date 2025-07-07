import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def generate_orientations(n_orientations, gamma_fixed=0.0, seed=None):
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
    weights_list = []

    for beta, w in zip(beta_vals, weights):
        n_alpha = max(1, int(round(w * n_orientations)))
        alpha_offset = rng.uniform(0, 2 * np.pi)
        alpha_vals = np.linspace(0, 2 * np.pi, n_alpha, endpoint=False) + alpha_offset
        alpha_vals %= 2 * np.pi

        for alpha in alpha_vals:
            angle_list.append([alpha, beta, gamma_fixed])

    return np.array(angle_list)

def visualize_orientations(orientations):
    alpha = np.degrees(orientations[:, 0])
    beta = np.degrees(orientations[:, 1])

    a_rad = np.radians(alpha)
    b_rad = np.radians(beta)
    x = np.sin(b_rad) * np.cos(a_rad)
    y = np.sin(b_rad) * np.sin(a_rad)
    z = np.cos(b_rad)
    points = np.vstack((x, y, z)).T

    hull = ConvexHull(points)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(alpha, beta, color='black', s=8)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, 180)
    ax1.set_xlabel(r'$\alpha$ (deg)')
    ax1.set_ylabel(r'$\beta$ (deg)')
    ax1.set_title('Euler Angles (α, β)')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for simplex in hull.simplices:
        ax2.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], color='black', linewidth=0.5)

    ax2.set_box_aspect([1, 1, 1])
    ax2.view_init(elev=20, azim=45)
    ax2.set_title('Triangulated Sphere Surface')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    orientations = generate_orientations(n_orientations=150)

    # Uncomment the next line to visualize when you want
    visualize_orientations(orientations)