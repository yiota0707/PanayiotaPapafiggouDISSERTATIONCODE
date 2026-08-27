# analyses how quickly cooperation develops and stabilises during training
# compares baseline and hybrid learning curves and convergence times across assortativity conditions

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# dissertation figure colours

BASELINE_COLOUR = "#7b6fd6"
HYBRID_COLOUR = "#c06082"


# file patterns

OUTCOME_FILE_PATTERN = re.compile(
    r"CountOutcomeT-sim(?P<simulation>\d+)\.txt$"
)
MODEL_PATTERN = re.compile(
    r"(?:^|[_-])(?P<model>baseline|hybrid)(?:[_-]|$)"
)
M_PATTERN = re.compile(
    r"(?:^|_)m(?P<m>\d+(?:\.\d+)?)"
)


# command line settings

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and hybrid learning speed."
    )

    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="Root folder containing result directories.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("learning_speed_analysis"),
        help="Directory used for analysis outputs.",
    )
    parser.add_argument(
        "--m-values", nargs="+", type=float, default=None,
        help="Optional assortativity values to analyse.",
    )
    parser.add_argument(
        "--smoothing-window", type=int, default=2000,
        help="Rolling window used to smooth cooperation.",
    )
    parser.add_argument(
        "--final-fraction", type=float, default=0.10,
        help="Final fraction used to estimate stable cooperation.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.03,
        help="Maximum difference from final cooperation.",
    )
    parser.add_argument(
        "--slope-window", type=int, default=2000,
        help="Window used to measure local cooperation change.",
    )
    parser.add_argument(
        "--slope-threshold", type=float, default=0.005,
        help="Maximum local change allowed for stability.",
    )
    parser.add_argument(
        "--persistence", type=int, default=3000,
        help="Number of consecutive stable iterations required.",
    )
    parser.add_argument(
        "--minimum-step", type=int, default=1000,
        help="Earliest iteration at which convergence can be detected.",
    )

    return parser.parse_args()


# load outcome data

def load_data(path: Path) -> np.ndarray:
    try:
        outcomes = np.loadtxt(path, delimiter=",")
    except ValueError:
        outcomes = np.loadtxt(path)

    outcomes = np.asarray(outcomes, dtype=np.float64)

    if outcomes.ndim == 1:
        outcomes = outcomes.reshape(1, -1)

    if outcomes.ndim != 2:
        raise ValueError(f"{path} is not a two-dimensional data file.")

    if outcomes.shape[1] != 4:
        raise ValueError(
            f"{path} should contain four columns [CC, CD, DC, DD]."
        )

    if np.any(outcomes < 0):
        raise ValueError(f"{path} contains negative outcome counts.")

    return outcomes


# calculate cooperation

def get_cooperation(outcomes: np.ndarray) -> np.ndarray:
    cc, cd, dc, dd = outcomes.T

    cooperative_actions = 2.0 * cc + cd + dc
    total_actions = 2.0 * (cc + cd + dc + dd)

    if np.any(total_actions == 0):
        raise ValueError(
            "At least one training iteration contains no outcomes."
        )

    return cooperative_actions / total_actions


# smooth cooperation curve

def smooth_curve(
    cooperation: np.ndarray,
    window: int,
) -> np.ndarray:
    if window < 1:
        raise ValueError("The smoothing window must be at least 1.")

    window = min(window, cooperation.size)

    return (
        pd.Series(cooperation)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


# measure local change

def get_local_change(
    smoothed_cooperation: np.ndarray,
    slope_window: int,
) -> np.ndarray:
    if slope_window < 1:
        raise ValueError("The slope window must be at least 1.")

    n_steps = smoothed_cooperation.size

    if n_steps == 1:
        return np.zeros(1, dtype=np.float64)

    slope_window = min(slope_window, n_steps - 1)

    differences = np.abs(
        smoothed_cooperation[slope_window:]
        - smoothed_cooperation[:-slope_window]
    )

    differences_per_1000 = differences * (1000.0 / slope_window)

    local_change = np.zeros(n_steps, dtype=np.float64)
    local_change[slope_window:] = differences_per_1000
    local_change[:slope_window] = differences_per_1000[0]

    return local_change


# find the first persistent stable period

def find_stable_point(
    stable_mask: np.ndarray,
    persistence: int,
) -> Optional[int]:
    if persistence < 1:
        raise ValueError("Persistence must be at least 1.")

    if stable_mask.size < persistence:
        return None

    stable_counts = np.convolve(
        stable_mask.astype(np.int64),
        np.ones(persistence, dtype=np.int64),
        mode="valid",
    )

    matches = np.flatnonzero(stable_counts == persistence)

    if matches.size == 0:
        return None

    return int(matches[0])


# detect convergence

def get_convergence(
    cooperation: np.ndarray,
    smoothing_window: int,
    final_fraction: float,
    tolerance: float,
    slope_window: int,
    slope_threshold: float,
    persistence: int,
    minimum_step: int,
) -> dict[str, object]:
    if not 0 < final_fraction <= 1:
        raise ValueError("final_fraction must be between 0 and 1.")

    if tolerance < 0:
        raise ValueError("tolerance cannot be negative.")

    if slope_threshold < 0:
        raise ValueError("slope_threshold cannot be negative.")

    smoothed = smooth_curve(cooperation, smoothing_window)
    local_change = get_local_change(smoothed, slope_window)

    n_steps = cooperation.size

    final_window = max(
        1,
        int(math.ceil(n_steps * final_fraction)),
    )

    final_values = cooperation[-final_window:]
    final_cooperation = float(np.mean(final_values))

    final_cooperation_sd = (
        float(np.std(final_values, ddof=1))
        if final_window > 1
        else 0.0
    )

    close_to_final = (
        np.abs(smoothed - final_cooperation)
        <= tolerance
    )

    low_change = (
        local_change
        <= slope_threshold
    )

    stable_mask = close_to_final & low_change

    if minimum_step > 1:
        stable_mask[
            : min(minimum_step - 1, stable_mask.size)
        ] = False

    convergence_index = find_stable_point(
        stable_mask,
        persistence,
    )

    if convergence_index is None:
        converged = False
        convergence_step = np.nan
        convergence_cooperation = np.nan
    else:
        converged = True
        convergence_step = float(convergence_index + 1)
        convergence_cooperation = float(
            smoothed[convergence_index]
        )

    return {
        "smoothed_cooperation": smoothed,
        "local_change": local_change,
        "stable_mask": stable_mask,
        "final_cooperation": final_cooperation,
        "final_cooperation_sd": final_cooperation_sd,
        "converged": converged,
        "convergence_step": convergence_step,
        "convergence_cooperation": convergence_cooperation,
    }


# find main experiment outcome files

def find_outcome_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(
            f"The supplied root directory does not exist: {root}"
        )

    excluded_terms = (
        "ablation",
        "component",
        "switch",
        "population",
        "horizon",
        "tau",
        "temperature",
        "learning_rate",
        "lr_ablation",
        "archive",
        "backup",
        "__pycache__",
    )

    selected_files = []

    for outcome_path in root.rglob("CountOutcomeT-sim*.txt"):
        if not OUTCOME_FILE_PATTERN.fullmatch(outcome_path.name):
            continue

        path_text = str(outcome_path).lower()

        if any(term in path_text for term in excluded_terms):
            continue

        result_directory = None

        for parent in outcome_path.parents:
            name = parent.name

            main_settings = (
                "_lr0.05_" in name
                and "_Nagent20_" in name
                and "_R20_" in name
            )

            baseline_match = (
                "baseline" in name
                and main_settings
                and (
                    name.endswith("_baseline_m0.00")
                    or name.endswith("_baseline_m1.00")
                )
            )

            hybrid_match = (
                "hybrid" in name
                and main_settings
                and (
                    name.endswith("_hybrid_m0.00")
                    or name.endswith("_hybrid_m1.00")
                )
            )

            if baseline_match or hybrid_match:
                result_directory = parent
                break

        if result_directory is not None:
            selected_files.append(outcome_path)

    selected_files = sorted(set(selected_files))

    if not selected_files:
        raise FileNotFoundError(
            "Could not identify the main baseline and hybrid result files."
        )

    return selected_files


# read model and assortativity from folder names

def get_metadata(path: Path) -> dict[str, object]:
    model: Optional[str] = None
    m_value: Optional[float] = None
    result_directory: Optional[str] = None

    for parent in path.parents:
        name = parent.name

        if model is None:
            model_match = MODEL_PATTERN.search(name)

            if model_match is not None:
                model = model_match.group("model")
                result_directory = name

        if m_value is None:
            m_match = M_PATTERN.search(name)

            if m_match is not None:
                m_value = float(m_match.group("m"))

    return {
        "model": model,
        "m": m_value,
        "result_directory": result_directory,
    }


# calculate summary statistics

def get_summary_stats(
    values: np.ndarray,
) -> tuple[float, float, float, float]:
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean_value = float(np.mean(values))

    if values.size == 1:
        return mean_value, 0.0, mean_value, mean_value

    sd = float(np.std(values, ddof=1))
    ci_margin = 1.96 * sd / math.sqrt(values.size)

    return (
        mean_value,
        sd,
        mean_value - ci_margin,
        mean_value + ci_margin,
    )


# summarise convergence results

def summarise_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (model, m_value), group in results.groupby(
        ["model", "m"],
        dropna=False,
    ):
        convergence_steps = group[
            "convergence_step"
        ].to_numpy(dtype=np.float64)

        final_cooperation = group[
            "final_cooperation"
        ].to_numpy(dtype=np.float64)

        (
            convergence_mean,
            convergence_sd,
            convergence_ci_low,
            convergence_ci_high,
        ) = get_summary_stats(convergence_steps)

        (
            final_mean,
            final_sd,
            final_ci_low,
            final_ci_high,
        ) = get_summary_stats(final_cooperation)

        rows.append(
            {
                "model": model,
                "m": m_value,
                "number_simulations": int(group.shape[0]),
                "number_converged": int(group["converged"].sum()),
                "convergence_rate": float(group["converged"].mean()),
                "mean_convergence_step": convergence_mean,
                "sd_convergence_step": convergence_sd,
                "convergence_ci95_low": convergence_ci_low,
                "convergence_ci95_high": convergence_ci_high,
                "mean_final_cooperation": final_mean,
                "sd_final_cooperation": final_sd,
                "final_cooperation_ci95_low": final_ci_low,
                "final_cooperation_ci95_high": final_ci_high,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["m", "model"],
        na_position="last",
    )


# plot cooperation learning curves

def plot_learning_curves(
    curves: dict[tuple[str, float], list[np.ndarray]],
    output_directory: Path,
) -> None:
    m_values = sorted(
        {
            m_value
            for _, m_value in curves
            if m_value is not None
        }
    )

    model_colours = {
        "baseline": BASELINE_COLOUR,
        "hybrid": HYBRID_COLOUR,
    }

    for m_value in m_values:
        models = [
            model
            for model in ("baseline", "hybrid")
            if (model, m_value) in curves
        ]

        if not models:
            continue

        minimum_length = min(
            min(curve.size for curve in curves[(model, m_value)])
            for model in models
        )

        steps = np.arange(1, minimum_length + 1)

        figure, axis = plt.subplots(figsize=(10, 6))

        for model in models:
            curve_matrix = np.vstack(
                [
                    curve[:minimum_length]
                    for curve in curves[(model, m_value)]
                ]
            )

            mean_curve = np.mean(curve_matrix, axis=0)

            if curve_matrix.shape[0] > 1:
                sd = np.std(
                    curve_matrix,
                    axis=0,
                    ddof=1,
                )

                ci = (
                    1.96
                    * sd
                    / math.sqrt(curve_matrix.shape[0])
                )
            else:
                ci = np.zeros(
                    minimum_length,
                    dtype=np.float64,
                )

            colour = model_colours[model]

            axis.plot(
                steps,
                mean_curve,
                color=colour,
                linewidth=2.5,
                label=model.capitalize(),
            )

            axis.fill_between(
                steps,
                mean_curve - ci,
                mean_curve + ci,
                color=colour,
                alpha=0.20,
            )

        axis.set_title(
            f"Cooperation Learning Speed: Baseline vs Hybrid, m={m_value:.2f}"
        )
        axis.set_xlabel("Training Iteration")
        axis.set_ylabel("Cooperation Rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()

        figure.tight_layout()

        figure.savefig(
            output_directory
            / f"learning_curves_m{m_value:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# plot convergence times

def plot_convergence(
    results: pd.DataFrame,
    output_directory: Path,
) -> None:
    converged_results = results[
        results["converged"]
    ].copy()

    if converged_results.empty:
        return

    m_values = sorted(
        converged_results["m"].dropna().unique()
    )

    model_colours = {
        "Baseline": BASELINE_COLOUR,
        "Hybrid": HYBRID_COLOUR,
    }

    for m_value in m_values:
        condition = converged_results[
            np.isclose(
                converged_results["m"],
                m_value,
            )
        ]

        convergence_data = []
        labels = []

        for model in ("baseline", "hybrid"):
            steps = (
                condition.loc[
                    condition["model"] == model,
                    "convergence_step",
                ]
                .dropna()
                .to_numpy(dtype=np.float64)
            )

            if steps.size > 0:
                convergence_data.append(steps)
                labels.append(model.capitalize())

        if len(convergence_data) < 2:
            continue

        figure, axis = plt.subplots(figsize=(7, 6))

        boxplot = axis.boxplot(
            convergence_data,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 7,
            },
            medianprops={
                "color": "black",
                "linewidth": 2,
            },
        )

        for patch, label in zip(
            boxplot["boxes"],
            labels,
        ):
            patch.set_facecolor(model_colours[label])
            patch.set_edgecolor("black")
            patch.set_alpha(0.85)

        axis.set_title(
            f"Time to Stable Cooperation, m={m_value:.2f}"
        )
        axis.set_ylabel("Convergence Iteration")
        axis.grid(axis="y", alpha=0.25)

        figure.tight_layout()

        figure.savefig(
            output_directory
            / f"convergence_comparison_m{m_value:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# run learning speed analysis

def run_analysis(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    output_directory = args.output.resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    outcome_files = find_outcome_files(root)

    rows = []
    curves: dict[
        tuple[str, float],
        list[np.ndarray],
    ] = {}

    for outcome_path in outcome_files:
        file_match = OUTCOME_FILE_PATTERN.fullmatch(
            outcome_path.name
        )

        if file_match is None:
            continue

        run_info = get_metadata(outcome_path)

        model = run_info["model"]
        m_value = run_info["m"]

        if model not in ("baseline", "hybrid"):
            continue

        if args.m_values is not None:
            if m_value is None:
                continue

            if not any(
                np.isclose(m_value, requested)
                for requested in args.m_values
            ):
                continue

        try:
            outcomes = load_data(outcome_path)
            cooperation = get_cooperation(outcomes)

            convergence = get_convergence(
                cooperation,
                args.smoothing_window,
                args.final_fraction,
                args.tolerance,
                args.slope_window,
                args.slope_threshold,
                args.persistence,
                args.minimum_step,
            )

        except (
            OSError,
            ValueError,
        ) as error:
            print(
                f"Skipping {outcome_path}: {error}"
            )
            continue

        simulation = int(
            file_match.group("simulation")
        )

        rows.append(
            {
                "model": model,
                "m": m_value,
                "simulation": simulation,
                "training_steps": int(cooperation.size),
                "final_cooperation": convergence["final_cooperation"],
                "final_cooperation_within_run_sd":
                    convergence["final_cooperation_sd"],
                "converged": convergence["converged"],
                "convergence_step": convergence["convergence_step"],
                "convergence_cooperation":
                    convergence["convergence_cooperation"],
                "result_directory": run_info["result_directory"],
                "source_file": str(outcome_path),
            }
        )

        curves.setdefault(
            (model, m_value),
            [],
        ).append(
            convergence["smoothed_cooperation"]
        )

        convergence_text = (
            f"{convergence['convergence_step']:.0f}"
            if convergence["converged"]
            else "not detected"
        )

        print(
            f"{model:8s} "
            f"m={m_value} "
            f"sim={simulation:04d} "
            f"convergence={convergence_text} "
            f"cooperation={convergence['final_cooperation']:.4f}"
        )

    if not rows:
        raise RuntimeError(
            "No valid baseline or hybrid results were analysed."
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            ["m", "model", "simulation"],
            na_position="last",
        )
    )

    results.to_csv(
        output_directory
        / "learning_speed_by_simulation.csv",
        index=False,
    )

    summary = summarise_results(results)

    summary.to_csv(
        output_directory
        / "learning_speed_summary.csv",
        index=False,
    )

    plot_learning_curves(
        curves,
        output_directory,
    )

    plot_convergence(
        results,
        output_directory,
    )

    print("\nLearning speed analysis completed.")
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    args = get_args()
    run_analysis(args)