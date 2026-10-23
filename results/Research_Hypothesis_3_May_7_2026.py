from pathlib import Path
"""Research Hypothesis 3: robustness to outliers.

Hypothesis: with outliers present, the robust method should recover the target
subspace better than the non-robust online method.
"""

import matplotlib.pyplot as plt

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np

from LokerePraxis import cosine_similarity, dorkPCA, grassmann_distance

SEED = 42
D = 30
D_SUBSPACE = 3
N_SAMPLES = 10_000
NOISE_STD = 0.1
EIGENVALUES = np.array([10, 3, 1])
OUTLIER_PROBABILITY = 0.05
OUTLIER_MAGNITUDE = 100
REPORT_STEP = 100

np.set_printoptions(suppress=True, precision=2, threshold=20)


def generate_data():
    rng = np.random.default_rng(SEED)
    mean = rng.uniform(0, 1, D)
    true_basis, _ = np.linalg.qr(rng.normal(0, 1, (D, D_SUBSPACE)))
    covariance = true_basis @ np.diag(EIGENVALUES) @ true_basis.T

    x = rng.multivariate_normal(mean, covariance, N_SAMPLES)
    x += rng.normal(0, NOISE_STD, x.shape)

    for sample in x:
        if rng.uniform() < OUTLIER_PROBABILITY:
            direction = rng.normal(0, 1, D)
            direction /= np.linalg.norm(direction)
            magnitude = rng.normal(OUTLIER_MAGNITUDE, 0.1 * OUTLIER_MAGNITUDE)
            sample += direction * magnitude

    return x, true_basis


def fit_and_track(model, x, true_basis, label):
    n = len(x)
    cos_sim = np.zeros((n, D_SUBSPACE))
    distances = np.zeros(n)

    for i, sample in enumerate(x):
        model.fit(sample)

        for j in range(D_SUBSPACE):
            cos_sim[i, j] = cosine_similarity(model.Gamma_.T[j], true_basis.T[j])
        distances[i] = grassmann_distance(true_basis, model.Gamma_)

        if i % REPORT_STEP == 0:
            print(
                f"\r{label} iteration {i}: distance: {distances[i]:.4f}",
                end="",
                flush=True,
            )

    print("\nfinal grassmann distance:")
    print(grassmann_distance(true_basis, model.Gamma_))
    print("\nestimated eigenvalues:")
    print(np.sqrt(model.evals_))

    return cos_sim, distances


def plot_run(cos_sim, distances, title):
    for j in range(D_SUBSPACE):
        plt.plot(cos_sim[:, j], label=f"Gamma[{j}] convergence")
    plt.plot(distances, label="Grassmann distance")
    plt.title(title)
    plt.legend()
    safe_title = title.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(PLOTS_DIR / f"{safe_title}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison(non_robust_distances, robust_distances):
    plt.plot(non_robust_distances, label="Without Robustness")
    plt.plot(robust_distances, label="With Robustness")
    plt.title("Subspace Recovery Error of\n Online Algorithm Compared to Ground-Truth")
    plt.xlabel("samples")
    plt.ylabel("Grassmann distance")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "robust_vs_non_robust_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    x, true_basis = generate_data()

    model_without = dorkPCA(
        D=D,
        d=D_SUBSPACE,
        eta=0.03,
        mu_learning_factor=0.1,
        Gamma_learning_factor=1,
        evals_learning_factor=0.1,
        rotation_factor=0.02,
        Psitype=0,
    )
    cos_without, distances_without = fit_and_track(
        model_without, x, true_basis, "non-robust"
    )
    plot_run(cos_without, distances_without, "Without robustness")

    model_with = dorkPCA(
        D=D,
        d=D_SUBSPACE,
        eta=0.1,
        mu_learning_factor=0.1,
        Gamma_learning_factor=1,
        evals_learning_factor=0.1,
        rotation_factor=0.02,
        Psitype=2,
        beta=0.8,
        ksi=0.1,
    )
    cos_with, distances_with = fit_and_track(
        model_with, x, true_basis, "robust"
    )
    plot_run(cos_with, distances_with, "With robustness")

    plot_comparison(distances_without, distances_with)


if __name__ == "__main__":
    main()
