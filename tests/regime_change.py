
"""Reproducible X -> Y -> Z regime-shift benchmark for dorkPCA.

    uv run python -m tests.regime_change --D 10 20 50 --d 2 5 10 --trials 50
    uv run python -m tests.regime_change --D 10 --d 2 --trials 1 --trajectory-trials 1

A trial contains three regimes of ``n_per_regime`` samples each. Both models
are evaluated against the same ground-truth bases. The global PCA baseline is
fit retrospectively to the entire stream (it is NOT an online baseline).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from LokerePraxis import dorkPCA, grassmann_distance

REGIMES = ("X", "Y", "Z")


@dataclass(frozen=True)
class ExperimentConfig:
    D: int
    d: int
    seed: int
    n_per_regime: int = 1000
    noise: float = 0.1
    threshold: float = 0.1
    consecutive: int = 25
    eigenvalues: tuple[float, ...] | None = None

    def validate(self) -> None:
        if not 1 <= self.d < self.D:
            raise ValueError(f"Require 1 <= d < D, got (D={self.D}, d={self.d})")
        if self.n_per_regime <= 0:
            raise ValueError("n_per_regime must be positive")
        if self.noise < 0:
            raise ValueError("noise must be nonnegative")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.consecutive <= 0:
            raise ValueError("consecutive must be positive")
        if self.eigenvalues is not None and (
            len(self.eigenvalues) != self.d or any(x <= 0 for x in self.eigenvalues)
        ):
            raise ValueError("eigenvalues must contain exactly d positive values")


def random_basis(D: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an orthonormal D x d basis."""
    q, _ = np.linalg.qr(rng.normal(size=(D, d)))
    return q[:, :d]


def sample_regime(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    n: int,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample N(0, B diag(eigenvalues) B.T + noise**2 I)."""
    z = rng.normal(size=(n, basis.shape[1])) * np.sqrt(eigenvalues)
    return z @ basis.T + noise * rng.normal(size=(n, basis.shape[0]))


def _first_sustained_crossing(errors: np.ndarray, threshold: float, consecutive: int):
    """1-based sample index first starting a full run below threshold; NA if none.

    For short regimes require at least ``consecutive`` observations. This avoids
    spuriously declaring convergence on a single brief threshold crossing.
    """
    if errors.size < consecutive:
        return pd.NA
    below = (errors < threshold).astype(np.int64)
    candidates = np.flatnonzero(np.convolve(below, np.ones(consecutive, dtype=int), mode="valid") == consecutive)
    return int(candidates[0] + 1) if candidates.size else pd.NA


def run_one_experiment(config: ExperimentConfig, collect_trajectory: bool = False):
    """Return (three regime-level metrics rows, optional sample-level trajectory).
 
    """
    config.validate()
    rng = np.random.default_rng(config.seed)
    eig = (np.linspace(5.0, 2.0, config.d)
           if config.eigenvalues is None
           else np.asarray(config.eigenvalues, dtype=float))

    bases = {r: random_basis(config.D, config.d, rng) for r in REGIMES}
    stream = np.vstack([
        sample_regime(bases[r], eig, config.n_per_regime, config.noise, rng)
        for r in REGIMES
    ])

    model = dorkPCA(D=config.D, d=config.d)
    errors = {r: np.empty(config.n_per_regime, dtype=float) for r in REGIMES}
    trajectory_rows = [] if collect_trajectory else None

    legacy_state = np.random.get_state()
    np.random.seed(config.seed)
    try:
        for i, x in enumerate(stream):
            model.fit(x)  # Current dorkPCA.fit() intentionally updates cumulatively.
            ri, k = divmod(i, config.n_per_regime)
            regime = REGIMES[ri]
            correct_distance = float(grassmann_distance(model.Gamma_, bases[regime]))
            errors[regime][k] = correct_distance
            if trajectory_rows is not None:
                trajectory_rows.append({
                    "seed": config.seed, "D": config.D, "d": config.d,
                    "t": i + 1, "regime": regime, "regime_t": k + 1,
                    "distance_X": float(grassmann_distance(model.Gamma_, bases["X"])),
                    "distance_Y": float(grassmann_distance(model.Gamma_, bases["Y"])),
                    "distance_Z": float(grassmann_distance(model.Gamma_, bases["Z"])),
                    "correct_distance": correct_distance,
                })
    finally:
        np.random.set_state(legacy_state)

    global_basis = PCA(n_components=config.d).fit(stream).components_.T
    batch_distances = {
        r: float(grassmann_distance(global_basis, bases[r])) for r in REGIMES
    }

    rows = []
    for regime in REGIMES:
        e = errors[regime]
        tail = e[-max(1, int(np.ceil(e.size / 4))):]
        recovery = _first_sustained_crossing(e, config.threshold, config.consecutive)
        rows.append({
            "seed": config.seed, "D": config.D, "d": config.d,
            "n_per_regime": config.n_per_regime,
            "noise": config.noise,
            "threshold": config.threshold,
            "consecutive": config.consecutive,
            "regime": regime,
            "initial_distance": float(e[0]),
            "final_distance": float(e[-1]),
            "mean_distance": float(np.mean(e)),
            "steady_state_mean": float(np.mean(tail)),
            "steady_state_median": float(np.median(tail)),
            "batch_pca_distance": batch_distances[regime],
            "paired_improvement": batch_distances[regime] - float(np.mean(tail)),
            "adaptation_time": recovery,
            "converged": not pd.isna(recovery),
        })

    return pd.DataFrame(rows), (pd.DataFrame(trajectory_rows) if trajectory_rows is not None else None)


def run_parameter_sweep(
    D_values: list[int],
    d_values: list[int],
    trials: int,
    *,
    base_seed: int = 42,
    n_per_regime: int = 1000,
    noise: float = 0.1,
    threshold: float = 0.1,
    consecutive: int = 25,
    trajectory_trials: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run independent seeded simulations; return metrics and optional trajectories.

    Seeds are unique per valid (D, d, trial) configuration and independent of
    loop-order changes, provided the set of requested combinations is unchanged.
    ``trajectory_trials`` retains detailed curves for the first K trials of
    each configuration; 0 disables trajectory recording entirely.
    """
    if trials <= 0 or trajectory_trials < 0:
        raise ValueError("trials must be positive and trajectory_trials nonnegative")
    configurations = [(D, d) for D in sorted(set(D_values))
                      for d in sorted(set(d_values)) if 1 <= d < D]
    if not configurations:
        raise ValueError("No valid D, d combinations (require 1 <= d < D)")
    seeds = np.random.SeedSequence(base_seed).spawn(len(configurations) * trials)
    metrics_parts, trajectory_parts = [], []
    index = 0
    for D, d in configurations:
        for trial in range(trials):
            seed = (base_seed if index == 0
                    else int(seeds[index].generate_state(1)[0]))
            index += 1
            config = ExperimentConfig(
                D=D, d=d, seed=seed, n_per_regime=n_per_regime,
                noise=noise, threshold=threshold, consecutive=consecutive,
            )
            m, path = run_one_experiment(config, collect_trajectory=(trial < trajectory_trials))
            m.insert(0, "trial", trial)
            metrics_parts.append(m)
            if path is not None:
                path.insert(0, "trial", trial)
                trajectory_parts.append(path)

        print(f"Finished D={D}, d={d}: {trials} trials", flush=True)

    metrics = pd.concat(metrics_parts, ignore_index=True)
    metrics["adaptation_time"] = metrics["adaptation_time"].astype("Int64")
    trajectories = (pd.concat(trajectory_parts, ignore_index=True)
                    if trajectory_parts else None)
    return metrics, trajectories


def summarize_experiments(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return regime-level summaries and independent trial-level summaries.

    Interquartile summaries represent run-to-run variability, not confidence
    intervals. Trial-level rows average the three correlated regimes so each
    simulation remains the unit of independent replication.
    """
    grouped = metrics.groupby(["D", "d", "regime"], sort=True)
    summary = grouped.agg(
        n_trials=("trial", "nunique"),
        rotation_median=("steady_state_mean", "median"),
        rotation_q25=("steady_state_mean", lambda x: x.quantile(0.25)),
        rotation_q75=("steady_state_mean", lambda x: x.quantile(0.75)),
        global_median=("batch_pca_distance", "median"),
        global_q25=("batch_pca_distance", lambda x: x.quantile(0.25)),
        global_q75=("batch_pca_distance", lambda x: x.quantile(0.75)),
        improvement_median=("paired_improvement", "median"),
        improvement_q25=("paired_improvement", lambda x: x.quantile(0.25)),
        improvement_q75=("paired_improvement", lambda x: x.quantile(0.75)),
        win_fraction=("paired_improvement", lambda x: float((x > 0).mean())),
        convergence_fraction=("converged", "mean"),
        recovery_median=("adaptation_time", "median"),
        recovery_q25=("adaptation_time", lambda x: x.quantile(.25)),
        recovery_q75=("adaptation_time", lambda x: x.quantile(.75)),
    ).reset_index()

    trial_summary = metrics.groupby(["D", "d", "trial", "seed"], sort=True).agg(
        rotation_error=("steady_state_mean", "mean"),
        global_error=("batch_pca_distance", "mean"),
        paired_improvement=("paired_improvement", "mean"),
        convergence_fraction=("converged", "mean"),
    ).reset_index()
    return summary, trial_summary


def summarize_trajectories(paths: pd.DataFrame | None) -> pd.DataFrame | None:
    if paths is None or paths.empty:
        return None
    grouped = paths.groupby(["D", "d", "t", "regime", "regime_t"], sort=True)
    return grouped["correct_distance"].agg(
        median="median",
        q25=lambda x: x.quantile(.25),
        q75=lambda x: x.quantile(.75),
        n_trials="size",
    ).reset_index()


def plot_comparison(summary: pd.DataFrame, plots_dir: Path) -> None:
    """One grouped comparison plot for each dimensional configuration."""
    for (D, d), df in summary.groupby(["D", "d"], sort=True):
        df = df.set_index("regime").loc[list(REGIMES)].reset_index()
        x = np.arange(len(df))
        width = .36
        fig, ax = plt.subplots(figsize=(8, 5))
        for offset, prefix, label in [
            (-width/2, "rotation", "Rotation-PCA (steady state)"),
            (+width/2, "global", "Global PCA (full stream)"),
        ]:
            med = df[f"{prefix}_median"].to_numpy(dtype=float)
            low = df[f"{prefix}_q25"].to_numpy(dtype=float)
            high = df[f"{prefix}_q75"].to_numpy(dtype=float)
            ax.bar(x + offset, med, width, label=label)
            ax.errorbar(x + offset, med,
                        yerr=np.vstack([np.maximum(med-low, 0), np.maximum(high-med, 0)]),
                        fmt="none", capsize=4, color="black", linewidth=1)
        ax.set_xticks(x, df["regime"])
        ax.set_xlabel("Regime")
        ax.set_ylabel("Grassmann distance")
        ax.set_title(f"Regime subspace error (D={D}, d={d}; median / IQR)")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(plots_dir / f"streaming_vs_batch_D{D}_d{d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_distances_to_each_basis(paths: pd.DataFrame | None, plots_dir: Path,
                                 n_per_regime: int) -> None:
    """Replicate the original three-basis plot; median/IQR for saved trials."""
    if paths is None:
        return
    for (D, d), subset in paths.groupby(["D", "d"], sort=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        for regime in REGIMES:
            grouped = subset.groupby("t")[f"distance_{regime}"]
            med = grouped.median()
            q25 = grouped.quantile(.25)
            q75 = grouped.quantile(.75)
            x = med.index.to_numpy()
            (line,) = ax.plot(x, med.to_numpy(), label=f"Distance to B_{regime}")
            ax.fill_between(x, q25.to_numpy(), q75.to_numpy(),
                            color=line.get_color(), alpha=.12)
        for change in (n_per_regime, 2*n_per_regime):
            ax.axvline(change + .5, color="grey", ls="--", alpha=.7)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Grassmann distance")
        ax.set_title(f"Subspace tracking by regime (D={D}, d={d})")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(plots_dir / f"regime_distances_D{D}_d{d}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)


def plot_tracking(paths_summary: pd.DataFrame | None, plots_dir: Path,
                  n_per_regime: int) -> None:
    """Median correct-basis distance with 25th–75th percentile ribbons."""
    if paths_summary is None:
        return
    for (D, d), df in paths_summary.groupby(["D", "d"], sort=True):
        df = df.sort_values("t")
        fig, ax = plt.subplots(figsize=(10, 5))
        t = df["t"].to_numpy()
        ax.plot(t, df["median"].to_numpy(), label="Median correct-basis distance")
        ax.fill_between(t, df["q25"].to_numpy(), df["q75"].to_numpy(),
                        alpha=.2, label="IQR")
        for change in (n_per_regime, 2*n_per_regime):
            ax.axvline(change + .5, color="grey", ls="--", alpha=.7)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Grassmann distance")
        count = int(df["n_trials"].max())
        ax.set_title(f"Tracking across regimes (D={D}, d={d}; {count} saved trials)")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(plots_dir / f"tracking_D{D}_d{d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--D", type=int, nargs="+", default=[10], help="Ambient dimensions")
    parser.add_argument("--d", type=int, nargs="+", default=[2], help="Target ranks")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Base seed for independent trials")
    parser.add_argument("--n-per-regime", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--consecutive", type=int, default=25)
    parser.add_argument("--trajectory-trials", type=int, default=1,
                        help="Save curves for first K trials per (D,d); 0 disables curves")
    parser.add_argument("--results-dir", type=Path, default=Path("results/regime_sweep"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots/regime_sweep"))
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    metrics, paths = run_parameter_sweep(
        D_values=args.D, d_values=args.d, trials=args.trials,
        base_seed=args.seed, n_per_regime=args.n_per_regime,
        noise=args.noise, threshold=args.threshold,
        consecutive=args.consecutive,
        trajectory_trials=args.trajectory_trials,
    )
    summary, trial_summary = summarize_experiments(metrics)
    path_summary = summarize_trajectories(paths)

    metrics.to_csv(args.results_dir / "regime_metrics.csv", index=False)
    trial_summary.to_csv(args.results_dir / "trial_metrics.csv", index=False)
    summary.to_csv(args.results_dir / "regime_summary.csv", index=False)
    if paths is not None:
        paths.to_csv(args.results_dir / "trajectories.csv", index=False)
    if path_summary is not None:
        path_summary.to_csv(args.results_dir / "trajectory_summary.csv", index=False)
    plot_comparison(summary, args.plots_dir)
    plot_tracking(path_summary, args.plots_dir, args.n_per_regime)
    plot_distances_to_each_basis(paths, args.plots_dir, args.n_per_regime)
    print("\nSummary (median and IQR across independent trials):")
    print(summary.to_string(index=False))
    print(f"\nOutputs: {args.results_dir} and {args.plots_dir}")


if __name__ == "__main__":
    main()