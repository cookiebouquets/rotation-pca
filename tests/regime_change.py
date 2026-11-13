"""X -> Y -> Z regime-change experiment, extended with parameter sweeps.

From repo root:
    uv run python -m tests.regime_change --D 10 20 50 --d 2 5 --trials 25
    uv run python -m tests.regime_change --D 10 --d 2 --trials 1
    uv run python -m tests.regime_change --D 20 50 --d 5 10 --eta .05 .1 .2 --trials 50

`--trajectory-trials` controls how many trials per setting retain per-sample
results for IQR tracking plots; all trials contribute to the metric summaries.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from LokerePraxis import dorkPCA, grassmann_distance

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def random_basis(D, d, rng):
    """Generate a random D x d orthonormal basis."""
    A = rng.normal(size=(D, d))
    Q, _ = np.linalg.qr(A)
    return Q[:, :d]


def sample_regime(B, eigenvalues, n, noise=0.1, rng=None):
    """Sample observations whose dominant covariance subspace is span(B)."""
    if rng is None:
        rng = np.random.default_rng()
    D, d = B.shape
    Z = rng.normal(size=(n, d))
    Z *= np.sqrt(eigenvalues)
    X_signal = Z @ B.T
    X_noise = noise * rng.normal(size=(n, D))
    return X_signal + X_noise


def run_one_experiment(D, d, seed=42, n_per_regime=1000, noise=0.1,
                       threshold=0.1, keep_trajectory=False, eta=0.1):
    """Return original per-regime metrics and optionally the samplewise results.

    fit() is cumulative in the current dorkPCA implementation, as in the
    original experiment. The estimator is newly instantiated for each trial.
    """
    if not 1 <= d < D:
        raise ValueError("Require 1 <= d < D")
    if n_per_regime < 1 or noise < 0:
        raise ValueError("n_per_regime must be positive and noise nonnegative")

    rng = np.random.default_rng(seed)
    eigenvalues = np.linspace(5.0, 2.0, d)

    B_X = random_basis(D, d, rng)
    B_Y = random_basis(D, d, rng)
    B_Z = random_basis(D, d, rng)
    X = sample_regime(B_X, eigenvalues, n_per_regime, noise, rng)
    Y = sample_regime(B_Y, eigenvalues, n_per_regime, noise, rng)
    Z = sample_regime(B_Z, eigenvalues, n_per_regime, noise, rng)
    stream = np.vstack([X, Y, Z])
    model = dorkPCA(D=D, d=d, eta=eta)

    records = []
    # The estimator's rotation machinery may use the legacy NumPy RNG.
    # Seed and restore it to keep trials reproducible and independent.
    old_state = np.random.get_state()
    np.random.seed(seed)
    try:
        for i, x in enumerate(stream):
            model.fit(x)
            if i < n_per_regime:
                regime, correct_basis = "X", B_X
            elif i < 2 * n_per_regime:
                regime, correct_basis = "Y", B_Y
            else:
                regime, correct_basis = "Z", B_Z

            # Preserve the original result columns and tracking calculation.
            if keep_trajectory:
                records.append({
                    "t": i,
                    "regime": regime,
                    "distance_X": grassmann_distance(model.Gamma_, B_X),
                    "distance_Y": grassmann_distance(model.Gamma_, B_Y),
                    "distance_Z": grassmann_distance(model.Gamma_, B_Z),
                    "correct_distance": grassmann_distance(model.Gamma_, correct_basis),
                })
            else:
                records.append({
                    "t": i,
                    "regime": regime,
                    "correct_distance": grassmann_distance(model.Gamma_, correct_basis),
                })
    finally:
        np.random.set_state(old_state)

    results = pd.DataFrame(records)
    results["regime_t"] = results.groupby("regime").cumcount() + 1

    # Preserve the original definitions: last quarter and first threshold crossing.
    metrics = []
    for regime in ["X", "Y", "Z"]:
        df = results[results["regime"] == regime]
        steady_state = df.tail(max(1, len(df) // 4))
        below_threshold = df[df["correct_distance"] < threshold]
        adaptation_time = (int(below_threshold["regime_t"].iloc[0])
                           if not below_threshold.empty else np.nan)
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

    batch_PCA = PCA(n_components=d)
    batch_PCA.fit(stream)
    B_batch = batch_PCA.components_.T
    batch_metrics = pd.DataFrame({
        "regime": ["X", "Y", "Z"],
        "batch_pca_distance": [
            grassmann_distance(B_batch, B_X),
            grassmann_distance(B_batch, B_Y),
            grassmann_distance(B_batch, B_Z),
        ],
    })
    metrics_df = metrics_df.merge(batch_metrics, on="regime")
    metrics_df["paired_improvement"] = (metrics_df["batch_pca_distance"]
                                        - metrics_df["steady_state_mean"])
    metrics_df["converged"] = metrics_df["adaptation_time"].notna()
    return metrics_df, (results if keep_trajectory else None)


def summarize_sweep(metrics_df):
    """Median and IQR over independent trials for each D/d/regime."""
    values = ["steady_state_mean", "batch_pca_distance", "paired_improvement",
              "adaptation_time"]
    grouped = metrics_df.groupby(["D", "d", "eta", "regime"])
    summary = grouped.size().rename("n_trials").reset_index()
    for value in values:
        stats = grouped[value].agg(
            **{f"{value}_median": "median",
               f"{value}_q25": lambda x: x.quantile(.25),
               f"{value}_q75": lambda x: x.quantile(.75),
            }
        ).reset_index()
        summary = summary.merge(stats, on=["D", "d", "eta", "regime"])
    success = grouped["converged"].mean().rename("convergence_rate").reset_index()
    summary = summary.merge(success, on=["D", "d", "eta", "regime"])
    return summary


def plot_sweep(summary, trajectories, plots_dir, n_per_regime):
    """Reuse original comparison/tracking plots with median and IQR summaries."""
    for (D, d, eta), df in summary.groupby(["D", "d", "eta"]):
        df = df.set_index("regime").loc[["X", "Y", "Z"]]
        x = np.arange(3)
        width = .35
        fig, ax = plt.subplots(figsize=(8, 5))
        for offset, column, label in [
            (-width / 2, "steady_state_mean", "Rotation-PCA steady-state error"),
            (+width / 2, "batch_pca_distance", "Global PCA distance"),
        ]:
            med = df[f"{column}_median"].to_numpy()
            q25 = df[f"{column}_q25"].to_numpy()
            q75 = df[f"{column}_q75"].to_numpy()
            ax.bar(x + offset, med, width, label=label)
            ax.errorbar(x + offset, med,
                        yerr=[np.maximum(0, med - q25), np.maximum(0, q75 - med)],
                        fmt="none", color="black", capsize=4)
        ax.set_xticks(x, ["X", "Y", "Z"])
        ax.set_xlabel("Regime")
        ax.set_ylabel("Grassmann distance")
        ax.set_title(f"Rotation-PCA vs Global PCA (D={D}, d={d}, eta={eta:g}; median/IQR)")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(plots_dir / f"streaming_vs_batch_D{D}_d{d}_eta{eta:g}.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if trajectories is None or trajectories.empty:
        return

    for (D, d, eta), df in trajectories.groupby(["D", "d", "eta"]):
        # Same two trajectory figures as original, now using median/IQR.
        for plot_name, columns, title in [
            ("regime_distances", ["distance_X", "distance_Y", "distance_Z"],
             "Subspace Tracking Across Regime Changes"),
            ("correct_basis_distance", ["correct_distance"],
             "Tracking Error Relative to Current Regime"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in columns:
                grouped = df.groupby("t")[col]
                med = grouped.median()
                q25, q75 = grouped.quantile(.25), grouped.quantile(.75)
                label = ("Distance to current true basis" if col == "correct_distance"
                         else f"Distance to B_{col[-1]}")
                line, = ax.plot(med.index, med.to_numpy(), label=label)
                ax.fill_between(med.index, q25.to_numpy(), q75.to_numpy(),
                                color=line.get_color(), alpha=.2)
            ax.axvline(n_per_regime, linestyle="--", alpha=.7)
            ax.axvline(2 * n_per_regime, linestyle="--", alpha=.7)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Grassmann distance")
            ax.set_title(f"{title} (D={D}, d={d}, eta={eta:g}; median/IQR)")
            ax.legend()
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{plot_name}_D{D}_d{d}_eta{eta:g}.png",
                        dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)


def plot_learning_rate_sweep(summary, plots_dir):
    """Plot steady-state error and recovery time against eta (median/IQR)."""
    for (D, d), df in summary.groupby(["D", "d"]):
        for value, ylabel, filename in [
            ("steady_state_mean", "Steady-state Grassmann distance", "error"),
            ("adaptation_time", "Samples to threshold (successful trials)", "adaptation"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            has_data = df[f"{value}_median"].notna().any()
            for regime, part in df.groupby("regime"):
                part = part.sort_values("eta")
                part = part.dropna(subset=[f"{value}_median"])
                if part.empty:
                    continue
                x = part["eta"].to_numpy()
                median = part[f"{value}_median"].to_numpy()
                q25 = part[f"{value}_q25"].to_numpy()
                q75 = part[f"{value}_q75"].to_numpy()
                line, = ax.plot(x, median, marker="o", label=f"Regime {regime}")
                ax.fill_between(x, q25, q75, color=line.get_color(), alpha=.2)
            if has_data:
                ax.set_xscale("log")
                ax.legend()
            else:
                ax.text(.5, .5, "No trials reached the threshold", ha="center",
                        va="center", transform=ax.transAxes)
            ax.set_xlabel("Overall learning rate (eta)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Learning-rate sweep (D={D}, d={d}; median/IQR)")
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(plots_dir / f"learning_rate_{filename}_D{D}_d{d}.png",
                        dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Rotation-PCA regime-shift sweep")
    parser.add_argument("--D", type=int, nargs="+", default=[10])
    parser.add_argument("--d", type=int, nargs="+", default=[2])
    parser.add_argument("--eta", type=float, nargs="+", default=[0.1],
                        help="Overall learning rates; does not scale rotation_factor")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-regime", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--trajectory-trials", type=int, default=1,
                        help="Number of trials per (D,d) to save for IQR time-series plots; 0 disables")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "regime_sweep")
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR / "regime_sweep")
    args = parser.parse_args()

    if any(not np.isfinite(eta) or eta <= 0 for eta in args.eta):
        parser.error("--eta values must be positive and finite")
    if args.trials < 1 or args.trajectory_trials < 0:
        parser.error("--trials must be positive and --trajectory-trials nonnegative")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    metrics, trajectories = [], []
    trial_id = 0
    for D in args.D:
        for d in args.d:
            if not 1 <= d < D:
                print(f"Skipping invalid configuration D={D}, d={d}")
                continue
            for trial in range(args.trials):
                # Same seed for each eta at a given (D, d, trial).
                # This keeps the generated stream identical for paired comparisons.
                seed = args.seed + trial_id
                trial_id += 1
                for eta in args.eta:
                    m, path = run_one_experiment(
                        D, d, seed=seed, n_per_regime=args.n_per_regime,
                        noise=args.noise, threshold=args.threshold,
                        keep_trajectory=(trial < args.trajectory_trials), eta=eta,
                    )
                    m.insert(0, "trial", trial)
                    m.insert(0, "seed", seed)
                    m.insert(0, "eta", eta)
                    m.insert(0, "d", d)
                    m.insert(0, "D", D)
                    metrics.append(m)
                    if path is not None:
                        path.insert(0, "trial", trial)
                        path.insert(0, "seed", seed)
                        path.insert(0, "eta", eta)
                        path.insert(0, "d", d)
                        path.insert(0, "D", D)
                        trajectories.append(path)
            print(f"Finished D={D}, d={d} ({args.trials} trials x {len(args.eta)} eta values)", flush=True)

    if not metrics:
        parser.error("No valid (D,d) configurations")
    metrics_df = pd.concat(metrics, ignore_index=True)
    summary_df = summarize_sweep(metrics_df)
    metrics_df.to_csv(args.results_dir / "regime_metrics.csv", index=False)
    summary_df.to_csv(args.results_dir / "regime_summary.csv", index=False)

    # Pairing is at the independent trial level, rather than treating 3 regimes
    # from the same stream as independent replications.
    trial_df = metrics_df.groupby(["D", "d", "eta", "trial", "seed"], as_index=False).agg(
        rotation_error=("steady_state_mean", "mean"),
        global_error=("batch_pca_distance", "mean"),
        paired_improvement=("paired_improvement", "mean"),
    )
    trial_df.to_csv(args.results_dir / "trial_metrics.csv", index=False)

    trajectories_df = (pd.concat(trajectories, ignore_index=True)
                       if trajectories else None)
    if trajectories_df is not None:
        trajectories_df.to_csv(args.results_dir / "trajectories.csv", index=False)
        trajectory_summary = trajectories_df.groupby(
            ["D", "d", "eta", "t", "regime", "regime_t"], as_index=False
        )["correct_distance"].agg(
            median="median", q25=lambda x: x.quantile(.25),
            q75=lambda x: x.quantile(.75)
        ).reset_index()
        trajectory_summary.to_csv(args.results_dir / "trajectory_summary.csv", index=False)

    plot_sweep(summary_df, trajectories_df, args.plots_dir, args.n_per_regime)
    plot_learning_rate_sweep(summary_df, args.plots_dir)
    print(summary_df.to_string(index=False))
    print(f"\nSaved CSVs to {args.results_dir} and plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
