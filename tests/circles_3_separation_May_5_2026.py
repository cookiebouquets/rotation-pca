from pathlib import Path
"""Three-circle separation with batch and online kernel PCA.
"""

import matplotlib.pyplot as plt

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split

from LokerePraxis import (
    cosine_similarity,
    dorkPCA,
    grassmann_distance,
    var_grassmann_distance,
)

SEED = 0
RBF_GAMMA = 0.3
M = 200
D_SUBSPACE = 2
ETA = 0.25
REPORT_STEP = 100

np.set_printoptions(suppress=True, precision=2, threshold=50)


def make_circle(rng, n_samples, noise=0.1, radius=1.0, label=0):
    """Generate one noisy circle."""
    theta = rng.random(n_samples) * 2 * np.pi
    x = np.zeros((n_samples, 2))
    x[:, 0] = radius * np.cos(theta) + rng.standard_normal(n_samples) * noise
    x[:, 1] = radius * np.sin(theta) + rng.standard_normal(n_samples) * noise
    y = np.full(n_samples, label, dtype=float)
    return x, y


def generate_data():
    rng = np.random.default_rng(SEED)
    x1, y1 = make_circle(rng, 600, noise=0.01, radius=1.0, label=0)
    x2, y2 = make_circle(rng, 1200, noise=0.01, radius=2.8, label=1)
    x3, y3 = make_circle(rng, 2400, noise=0.01, radius=5.0, label=2)

    x = np.vstack([x1, x2, x3])
    y = np.hstack([y1, y2, y3])
    return train_test_split(x, y, stratify=y, random_state=SEED)


def rbf_kernel(x1, x2, gamma=1.0):
    squared_distance = np.sum((x1 - x2) ** 2)
    return np.exp(-gamma * squared_distance)


def kernel_features(x, basis, gamma=RBF_GAMMA):
    """Map samples to RBF similarities against the basis U."""
    features = np.zeros((len(x), len(basis)))
    for i, sample in enumerate(x):
        for j, basis_sample in enumerate(basis):
            features[i, j] = rbf_kernel(sample, basis_sample, gamma)
    return features


def batch_reference(x_train, x_test, rng):
    """Build the scikit-learn KPCA baseline and reduced-U batch reference."""
    kernel_pca = KernelPCA(
        n_components=None,
        kernel="rbf",
        gamma=RBF_GAMMA,
        fit_inverse_transform=True,
        alpha=0.1,
    )
    x_test_kernel_pca = kernel_pca.fit(x_train).transform(x_test)

    basis_indices = rng.choice(x_train.shape[0], M, replace=False)
    u = x_train[basis_indices]
    ku = kernel_features(x_train, u)
    ku_test = kernel_features(x_test, u)

    ku_mean = np.mean(ku, axis=0)
    eigenvalues, eigenvectors = np.linalg.eig((ku - ku_mean).T @ (ku - ku_mean))
    print("Dominant Eigenvalues:")
    print(np.round(np.real(eigenvalues), 2)[:25])

    reduced_eigenvectors = np.real(
        eigenvectors[:, :D_SUBSPACE] / np.sqrt(eigenvalues[:D_SUBSPACE])
    )
    x_test_u_kpca = (ku_test - ku_mean) @ reduced_eigenvectors
    normalized_eigenvectors = reduced_eigenvectors * np.sqrt(
        eigenvalues[:D_SUBSPACE]
    )

    return (
        x_test_kernel_pca,
        u,
        ku,
        ku_test,
        ku_mean,
        x_test_u_kpca,
        normalized_eigenvectors,
    )


def run_online(x_train, u, ku_mean, normalized_eigenvectors):
    n = len(x_train)
    cos_sim = np.zeros((n, D_SUBSPACE))
    cos_sim_mu = np.zeros(n)
    distances = np.zeros(n)
    var_distances = np.zeros(n)

    model = dorkPCA(
        D=2,
        d=D_SUBSPACE,
        eta=ETA,
        U=u,
        kernel_function="rbf_kernel",
        rbf_gamma=RBF_GAMMA,
    )

    for i, sample in enumerate(x_train):
        model.fit(sample)
        for j in range(D_SUBSPACE):
            cos_sim[i, j] = cosine_similarity(
                model.Gamma_.T[j], normalized_eigenvectors.T[j]
            )
        cos_sim_mu[i] = cosine_similarity(model.mu_, ku_mean)
        distances[i] = grassmann_distance(model.Gamma_, normalized_eigenvectors)
        var_distances[i] = var_grassmann_distance(
            model.Gamma_, normalized_eigenvectors, model.evals_
        )

        if i % REPORT_STEP == 0:
            print(
                f"\riteration {i}: distance: {distances[i]:.4f} "
                f"var_distance: {var_distances[i]:.4f}",
                end="",
                flush=True,
            )

    print("\nfinal grassmann distance:")
    print(grassmann_distance(model.Gamma_, normalized_eigenvectors))
    print("\nestimated eigenvalues:")
    print(np.sqrt(model.evals_))

    return model, cos_sim, distances, var_distances


def plot_training_data(x_train, y_train):
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.xlabel("Feature #0")
    plt.ylabel("Feature #1")
    plt.title("Training data")
    plt.savefig(PLOTS_DIR / "training_data.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_convergence(cos_sim, distances, var_distances):
    for j in range(D_SUBSPACE):
        plt.plot(cos_sim[:, j], label=f"Gamma[{j}] convergence")
    plt.plot(distances, label="Grassmann distance")
    plt.plot(var_distances, label="Var Grassmann distance")
    plt.legend()
    plt.savefig(PLOTS_DIR / "convergence.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_projections(x_test, y_test, sklearn_proj, reduced_proj, online_proj):
    fig, axes = plt.subplots(ncols=4, figsize=(20, 4))
    panels = [
        (x_test, "Testing data", "Feature #0", "Feature #1"),
        (sklearn_proj, "Projection of testing data\n using KernelPCA", "Principal component #0", "Principal component #1"),
        (reduced_proj, "Projection using Batch KernelPCA \n with smaller U basis", "Principal component #0", "Principal component #1"),
        (online_proj, "Projection of testing data\n using Online KernelPCA", "Principal component #0", "Principal component #1"),
    ]
    for ax, (values, title, xlabel, ylabel) in zip(axes, panels):
        ax.scatter(values[:, 0], values[:, 1], c=y_test)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    plt.savefig(PLOTS_DIR / "projection_comparison_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
    panels = [
        (x_test, "Testing data - not separable", "Feature #0", "Feature #1"),
        (sklearn_proj, "Projection of data using\n Batch KernelPCA (Scikit Learn)", "Principal component #0", "Principal component #1"),
        (online_proj, "Projection of data using\n Proposed Online Kernel PCA", "Principal component #0", "Principal component #1"),
    ]
    for ax, (values, title, xlabel, ylabel) in zip(axes, panels):
        ax.scatter(values[:, 0], values[:, 1], c=y_test)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    plt.savefig(PLOTS_DIR / "projection_comparison_reduced.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    x_train, x_test, y_train, y_test = generate_data()
    plot_training_data(x_train, y_train)

    rng = np.random.default_rng(SEED)
    (
        x_test_kernel_pca,
        u,
        ku,
        ku_test,
        ku_mean,
        x_test_u_kpca,
        normalized_eigenvectors,
    ) = batch_reference(x_train, x_test, rng)

    print("Number of elements in U", len(u))
    print("Train data", ku.shape)
    print("Test data", ku_test.shape)
    print("Eigenvectors:", normalized_eigenvectors.shape)
    print("Batch Mean:", ku_mean.shape)

    model, cos_sim, distances, var_distances = run_online(
        x_train, u, ku_mean, normalized_eigenvectors
    )
    plot_convergence(cos_sim, distances, var_distances)

    # May 5 notebook computed online test projections explicitly from KU_test.
    x_test_okpca = (ku_test - model.mu_) @ model.Gamma_
    plot_projections(
        x_test,
        y_test,
        x_test_kernel_pca,
        x_test_u_kpca,
        x_test_okpca,
    )


if __name__ == "__main__":
    main()
