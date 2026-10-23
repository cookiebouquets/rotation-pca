from pathlib import Path
"""Convergence rate versus eigenvalue magnitude for Lokere online PCA.
"""

import warnings

import matplotlib.pyplot as plt

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np

from LokerePraxis import (
    cosine_similarity,
    dorkPCA,
    grassmann_distance,
    var_grassmann_distance,
)

SEED = 42
D = 100
D_SUBSPACE = 10
N_SAMPLES = 10_000
NOISE_STD = 0.1
EIGENVALUES = np.array([60, 20, 15, 8, 4, 0.1, 0.1, 0.1, 0.1, 0.1])
REPORT_STEP = 100


def generate_data():
    """Generate the Gaussian stream and return samples plus the true basis."""
    rng = np.random.default_rng(SEED)
    mean = rng.uniform(0, 1, D)
    true_basis, _ = np.linalg.qr(rng.normal(0, 1, (D, D_SUBSPACE)))
    covariance = true_basis @ np.diag(EIGENVALUES) @ true_basis.T
    x = rng.multivariate_normal(mean, covariance, N_SAMPLES)
    x += rng.normal(0, NOISE_STD, x.shape)
    return x, true_basis


def run_experiment(x, true_basis):
    """Fit the online model and record convergence metrics."""
    n = len(x)
    cos_sim = np.zeros((n, D_SUBSPACE))
    distances = np.zeros(n)
    var_distances = np.zeros(n)

    model = dorkPCA(D=D, d=D_SUBSPACE)

    for i, sample in enumerate(x):
        model.fit(sample)

        # The original experiment tracks the first five eigenvectors.
        for j in range(5):
            cos_sim[i, j] = cosine_similarity(model.Gamma_.T[j], true_basis.T[j])

        distances[i] = grassmann_distance(model.Gamma_, true_basis)
        var_distances[i] = var_grassmann_distance(
            model.Gamma_, true_basis, model.evals_
        )

        if i % REPORT_STEP == 0:
            print(
                f"\riteration {i}: distance: {distances[i]:.4f} "
                f"var_distance: {var_distances[i]:.4f}",
                end="",
                flush=True,
            )

    print()
    return cos_sim, distances, var_distances


def plot_convergence(cos_sim):
    """Plot cosine-similarity convergence for the five leading components."""
    warnings.filterwarnings("ignore")
    for j in range(5):
        plt.plot(cos_sim[:, j], label=rf"eigenvalue $\lambda=$ {EIGENVALUES[j]}")

    plt.title("Convergence of eigenvectors\n versus eigenvalue")
    plt.xlabel("Samples")
    plt.ylabel("Cosine similarity")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOTS_DIR / "convergence_vs_eigenvalues.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    x, true_basis = generate_data()
    cos_sim, _, _ = run_experiment(x, true_basis)
    plot_convergence(cos_sim)


if __name__ == "__main__":
    main()
