from pathlib import Path
"""Effect of rotation on variance-weighted subspace convergence.

"""

import matplotlib.pyplot as plt

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np

from LokerePraxis import dorkPCA, var_grassmann_distance

SEED = 42
D = 100
D_SUBSPACE = 10
N_SAMPLES = 20_000
NOISE_STD = 1.0
EIGENVALUES = np.array([100, 80, 64, 50, 40, 32, 25, 20, 16, 10])
REPORT_STEP = 50


def generate_data():
    rng = np.random.default_rng(SEED)
    mean = rng.uniform(0, 1, D)
    true_basis, _ = np.linalg.qr(rng.normal(0, 1, (D, D_SUBSPACE)))
    covariance = true_basis @ np.diag(EIGENVALUES) @ true_basis.T
    x = rng.multivariate_normal(mean, covariance, N_SAMPLES)
    x += rng.normal(0, NOISE_STD, x.shape)
    return x, true_basis


def run_experiment(x, true_basis):
    model_with = dorkPCA(D=D, d=D_SUBSPACE, rotation=True)
    model_without = dorkPCA(D=D, d=D_SUBSPACE, rotation=False)

    n = len(x)
    with_rotation = np.zeros(n)
    without_rotation = np.zeros(n)

    for i, sample in enumerate(x):
        model_with.fit(sample)
        model_without.fit(sample)

        with_rotation[i] = var_grassmann_distance(
            model_with.Gamma_, true_basis, model_with.evals_
        )
        without_rotation[i] = var_grassmann_distance(
            model_without.Gamma_, true_basis, model_without.evals_
        )

        if i % REPORT_STEP == 0:
            print(
                f"\riteration {i}: var_distance with: {with_rotation[i]:.4f} "
                f"without: {without_rotation[i]:.4f}",
                end="",
                flush=True,
            )

    print()
    return with_rotation, without_rotation


def plot_results(with_rotation, without_rotation):
    samples = np.arange(len(with_rotation))
    plt.figure(figsize=(10, 5))
    plt.plot(samples, without_rotation, label="without rotation")
    plt.plot(samples, with_rotation, label="with rotation")
    plt.title("Variance Weighted Subspace Convergence")
    plt.xlabel("Samples")
    plt.ylabel("Variance Weighted Grassmann Distance")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(PLOTS_DIR / "rotation_variance_weighted_grassmann_distance.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    x, true_basis = generate_data()
    with_rotation, without_rotation = run_experiment(x, true_basis)
    plot_results(with_rotation, without_rotation)


if __name__ == "__main__":
    main()
