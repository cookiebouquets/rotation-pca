"""
Research Question:

Suppose a data generating process can be decomposed into a cyclic progression of sub data generating processes, i.e. 

S = X -> Y -> Z -> X -> ....

Does a rotation based adaptive PCA perform better by repeatedly fitting and shifting between B_X, B_Y, and B_Z compared to fitting a global PCA basis, B_S?
"""


from pathlib import Path

import matplotlib.pyplot as plt
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents= True, exist_ok=True)
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from LokerePraxis import (
    cosine_similarity,
    dorkPCA,
    grassmann_distance,
    var_grassmann_distance,
)

def random_basis(D, d, rng):
    """Generate a random D x d orthonormal basis."""
    A = rng.normal(size=(D, d))
    Q, _ = np.linalg.qr(A)
    return Q[:, :d]


def sample_regime(B, eigenvalues, n, noise=0.1, rng=None):
    """
    Sample observations whose dominant covariance subspace is span(B).
    """
    if rng is None:
        rng = np.random.default_rng()

    D, d = B.shape

    # coordinates within the true low-dimensional subspace
    Z = rng.normal(size=(n, d))
    Z *= np.sqrt(eigenvalues)

    # project into ambient D-dimensional space
    X_signal = Z @ B.T

    # small isotropic noise outside the subspace
    X_noise = noise * rng.normal(size=(n, D))

    return X_signal + X_noise


def main():
    rng = np.random.default_rng(42)

    D = 10
    d = 2
    n_per_regime = 1000

    eigenvalues = np.array([5.0, 2.0])

    B_X = random_basis(D, d, rng)
    B_Y = random_basis(D, d, rng)
    B_Z = random_basis(D, d, rng)

    X = sample_regime(
        B_X,
        eigenvalues,
        n=n_per_regime,
        noise=0.1,
        rng=rng
    )

    Y = sample_regime(
        B_Y,
        eigenvalues,
        n=n_per_regime,
        noise=0.1,
        rng=rng
    )

    Z = sample_regime(
        B_Z,
        eigenvalues,
        n=n_per_regime,
        noise=0.1,
        rng=rng
    )

    stream = np.vstack([X, Y, Z])
    model = dorkPCA(D=D, d=d)

    records = []

    for i, x in enumerate(stream):
        model.fit(x)

        if i < n_per_regime:
            regime = "X"
            correct_basis = B_X
        elif i < 2 * n_per_regime:
            regime = "Y"
            correct_basis = B_Y
        else:
            regime = "Z"
            correct_basis = B_Z

        records.append({
            "t": i,
            "regime": regime,
            "distance_X": grassmann_distance(model.Gamma_, B_X),
            "distance_Y": grassmann_distance(model.Gamma_, B_Y),
            "distance_Z": grassmann_distance(model.Gamma_, B_Z),
            "correct_distance": grassmann_distance(
                model.Gamma_,
                correct_basis
            ),
        })

    results = pd.DataFrame(records)
    results["regime_t"] = results.groupby("regime").cumcount() + 1


    # 1. Distance to each regime basis over the full stream
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(results["t"], results["distance_X"], label="Distance to B_X")
    ax.plot(results["t"], results["distance_Y"], label="Distance to B_Y")
    ax.plot(results["t"], results["distance_Z"], label="Distance to B_Z")

    ax.axvline(n_per_regime, linestyle="--", alpha=0.7)
    ax.axvline(2 * n_per_regime, linestyle="--", alpha=0.7)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Grassmann distance")
    ax.set_title("Subspace Tracking Across Regime Changes")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "regime_distances.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)


    # 2. Distance to the currently correct regime basis
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        results["t"],
        results["correct_distance"],
        label="Distance to current true basis"
    )

    ax.axvline(n_per_regime, linestyle="--", alpha=0.7)
    ax.axvline(2 * n_per_regime, linestyle="--", alpha=0.7)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Grassmann distance")
    ax.set_title("Tracking Error Relative to Current Regime")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "correct_basis_distance.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)

    # Metrics

    threshold = 0.1
    print("Metrics")
    metrics = []

    for regime in ["X", "Y", "Z"]:
        df = results[results["regime"] == regime].copy()

        steady_state = df.tail(len(df) // 4)

        below_threshold = df[df["correct_distance"] < threshold]

        adaptation_time = (
            below_threshold["regime_t"].iloc[0]
            if not below_threshold.empty
            else None
        )

        metrics.append({
            "regime": regime,
            "initial_distance": df["correct_distance"].iloc[0],
            "final_distance": df["correct_distance"].iloc[-1],
            "mean_distance": df["correct_distance"].mean(),
            "steady_state_mean": steady_state["correct_distance"].mean(),
            "steady_state_median": steady_state["correct_distance"].median(),
            "adaptation_time": adaptation_time,
        })

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    metrics_df.to_csv(
        RESULTS_DIR / "regime_shift_metrics.csv",
        index=False
    )


    # Next, fit a global n_components = 2 basis onto our stream dataset

    batch_PCA = PCA(n_components = d)
    batch_PCA.fit(stream)

    B_batch = batch_PCA.components_.T
    dist_X = grassmann_distance(B_batch, B_X)
    dist_Y = grassmann_distance(B_batch, B_Y)
    dist_Z = grassmann_distance(B_batch, B_Z)

    print("Batch PCA distance to B_X:", dist_X)
    print("Batch PCA distance to B_Y:", dist_Y)
    print("Batch PCA distance to B_Z:", dist_Z)


    batch_metrics = pd.DataFrame({
        "regime": ["X", "Y", "Z"],
        "batch_pca_distance": [
            grassmann_distance(B_batch, B_X),
            grassmann_distance(B_batch, B_Y),
            grassmann_distance(B_batch, B_Z),
        ]
    })
    # Pull just the steady-state streaming error
    streaming_metrics = metrics_df[["regime", "steady_state_mean"]].copy()

    # Merge into one comparison table
    metrics_df = metrics_df.reset_index()
    comparison_df = streaming_metrics.merge(batch_metrics, on="regime")

    print(comparison_df)

    # Plot
    x = np.arange(len(comparison_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        comparison_df["steady_state_mean"],
        width,
        label="Rotation-PCA steady-state error"
    )

    ax.bar(
        x + width / 2,
        comparison_df["batch_pca_distance"],
        width,
        label="Static batch PCA distance"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["regime"])
    ax.set_xlabel("Regime")
    ax.set_ylabel("Grassmann distance")
    ax.set_title("Rotation-PCA vs Static PCA by Regime")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "streaming_vs_batch_pca.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)



if __name__ == "__main__":
    main()