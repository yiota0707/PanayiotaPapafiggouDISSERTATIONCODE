# runs and analyses the initial cooperation q-value bias experiment
# tests whether cooperative initialisation affects final cooperation, residual defection and learned strategies

from __future__ import annotations
import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# plot colours

PURPLE = "#7b6fd6"
PINK = "#c06082"

MODEL_COLOURS = {
    "baseline": PURPLE,
    "hybrid": PINK,
}

BIAS_LINE_STYLES = {
    0.0: "--",
    0.5: "-",
}

SIMULATION_PATTERN = re.compile(
    r"CountOutcomeT-sim(?P<simulation>\d+)\.txt$"
)


# command line arguments

def get_args():
    parser = argparse.ArgumentParser(
        description="Run and analyse the initial cooperation Q-value bias experiment."
    )

    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("MAgame2106s2_baseline_assort_m_njit.py"),
    )
    parser.add_argument(
        "--hybrid-file",
        type=Path,
        default=Path("MAgame2106s2_hybrid_assort_m_njit.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("initial_cooperation_bias_results"),
    )
    parser.add_argument(
        "--bias-values",
        nargs="+",
        type=float,
        default=[0.0, 0.5],
    )
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--nsim", type=int, default=10)
    parser.add_argument("--T", type=int, default=100000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--final-fraction", type=float, default=0.10)
    parser.add_argument("--defector-threshold", type=float, default=0.50)
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


# create temporary model with initial cooperation bias

def make_bias_model(source_path, temporary_path, cooperation_bias):
    source_path = source_path.resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Model file not found: {source_path}")

    source = source_path.read_text(encoding="utf-8")

    target = "Q = np.zeros((number_agents, 4, 2), dtype=np.float64)"

    if target not in source:
        raise RuntimeError(
            f"Could not find Q-table initialisation in {source_path.name}."
        )

    replacement = (
        f"{target}\n"
        f"    Q[:, 2, 0] = {cooperation_bias!r}\n"
        f"    Q[:, 3, 0] = {cooperation_bias!r}"
    )

    source = source.replace(target, replacement, 1)

    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(source, encoding="utf-8")


# get result directory for one condition

def get_result_dir(condition_directory, model, m_value):
    algorithm = (
        "Qpast1-b-baseline"
        if model == "baseline"
        else "Qpast1-b-hybrid"
    )

    return (
        condition_directory
        / (
            f"result_PD2_{algorithm}_lr0.05_"
            f"Npast1_Nagent20_R20_{model}_m{m_value:.2f}"
        )
    )


# count completed simulations

def count_sims(result_directory):
    if not result_directory.exists():
        return 0

    return len(
        list(result_directory.rglob("CountOutcomeT-sim*.txt"))
    )


# run one bias condition

def run_condition(source_file, model, bias, args):
    condition_directory = (
        args.output.resolve()
        / model
        / f"qinitC{bias:.2f}"
    )

    result_directory = get_result_dir(
        condition_directory,
        model,
        args.m,
    )

    if count_sims(result_directory) >= args.nsim and not args.force:
        return

    if args.force and condition_directory.exists():
        shutil.rmtree(condition_directory)

    condition_directory.mkdir(parents=True, exist_ok=True)

    temporary_script = (
        condition_directory
        / f"_temporary_{model}_qinitC{bias:.2f}.py"
    )

    make_bias_model(
        source_file,
        temporary_script,
        bias,
    )

    command = [
        sys.executable,
        temporary_script.name,
        str(args.m),
        str(args.lr),
        str(args.tau),
        str(args.nsim),
        str(args.T),
    ]

    log_path = (
        condition_directory
        / f"{model}_qinitC{bias:.2f}.log"
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=condition_directory,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    temporary_script.unlink(missing_ok=True)

    if process.returncode != 0:
        raise RuntimeError(
            f"{model} Qinit(C)={bias:.2f} failed. Check {log_path}."
        )


# load simulation data

def load_data(path):
    try:
        values = np.loadtxt(path, delimiter=",")
    except ValueError:
        values = np.loadtxt(path)

    values = np.asarray(values, dtype=np.float64)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    return values


# calculate cooperation from outcome counts

def get_cooperation(outcomes):
    if outcomes.ndim != 2 or outcomes.shape[1] != 4:
        raise ValueError(
            "CountOutcomeT must contain four columns: CC, CD, DC and DD."
        )

    cc, cd, dc, dd = outcomes.T

    cooperative_actions = 2.0 * cc + cd + dc
    total_actions = 2.0 * (cc + cd + dc + dd)

    return np.divide(
        cooperative_actions,
        total_actions,
        out=np.full_like(cooperative_actions, np.nan),
        where=total_actions > 0,
    )


# smooth learning curves

def moving_average(values, window):
    window = max(1, min(int(window), len(values)))

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


# find completed simulations

def find_sims(result_directory):
    simulations = []

    for path in result_directory.rglob("CountOutcomeT-sim*.txt"):
        match = SIMULATION_PATTERN.fullmatch(path.name)

        if match:
            simulations.append(
                (
                    int(match.group("simulation")),
                    path.parent,
                )
            )

    return sorted(simulations, key=lambda item: item[0])


# collect initial bias results

def get_results(args):
    rows = []
    learning_curves = {}

    for model in ["baseline", "hybrid"]:
        for bias in args.bias_values:
            bias = float(bias)

            condition_directory = (
                args.output.resolve()
                / model
                / f"qinitC{bias:.2f}"
            )

            result_directory = get_result_dir(
                condition_directory,
                model,
                args.m,
            )

            simulations = find_sims(result_directory)

            if not simulations:
                continue

            key = (model, bias)
            learning_curves.setdefault(key, [])

            for simulation, simulation_directory in simulations:
                outcomes_path = (
                    simulation_directory
                    / f"CountOutcomeT-sim{simulation:04d}.txt"
                )
                defections_path = (
                    simulation_directory
                    / f"AgentCountDT-sim{simulation:04d}.txt"
                )
                policy_path = (
                    simulation_directory
                    / f"CountPolicyPDTypeT-sim{simulation:04d}.txt"
                )

                if not outcomes_path.exists() or not defections_path.exists():
                    continue

                try:
                    outcomes = load_data(outcomes_path)
                    defections = load_data(defections_path)
                except (OSError, ValueError):
                    continue

                cooperation = get_cooperation(outcomes)

                n_rows = min(
                    len(cooperation),
                    defections.shape[0],
                )

                cooperation = cooperation[:n_rows]
                defections = defections[:n_rows]

                final_length = max(
                    1,
                    math.ceil(
                        n_rows * args.final_fraction
                    ),
                )

                final_slice = slice(
                    n_rows - final_length,
                    n_rows,
                )

                final_cooperation = float(
                    np.nanmean(
                        cooperation[final_slice]
                    )
                )

                final_defection_rates = (
                    np.mean(
                        defections[final_slice],
                        axis=0,
                    )
                    / float(args.rounds)
                )

                remaining_defectors = (
                    final_defection_rates
                    >= args.defector_threshold
                )

                population = defections.shape[1]
                remaining_count = int(
                    np.sum(remaining_defectors)
                )

                strategies = [np.nan] * 4

                if policy_path.exists():
                    policies = load_data(policy_path)
                    policy_rows = min(len(policies), n_rows)
                    policies = policies[:policy_rows]

                    policy_length = max(
                        1,
                        math.ceil(
                            policy_rows
                            * args.final_fraction
                        ),
                    )

                    totals = policies.sum(
                        axis=1,
                        keepdims=True,
                    )
                    totals[totals == 0] = 1.0

                    proportions = policies / totals

                    strategies = np.mean(
                        proportions[-policy_length:],
                        axis=0,
                    )

                rows.append(
                    {
                        "model": model,
                        "m": args.m,
                        "initial_cooperation_q": bias,
                        "simulation": simulation,
                        "population": population,
                        "final_cooperation": final_cooperation,
                        "final_noncooperation": 1.0 - final_cooperation,
                        "remaining_defector_count": remaining_count,
                        "remaining_defector_fraction":
                            remaining_count / float(population),
                        "pd_always_cooperate": strategies[0],
                        "pd_conditional": strategies[1],
                        "pd_contrarian": strategies[2],
                        "pd_always_defect": strategies[3],
                    }
                )

                learning_curves[key].append(cooperation)

    if not rows:
        raise RuntimeError(
            "No complete bias-experiment simulations were found."
        )

    results = pd.DataFrame(rows).sort_values(
        ["model", "initial_cooperation_q", "simulation"]
    )

    return results, learning_curves


# calculate confidence interval

def get_ci(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    mean_value = np.mean(values)

    if len(values) == 1:
        return mean_value, mean_value

    standard_error = (
        np.std(values, ddof=1)
        / np.sqrt(len(values))
    )

    margin = 1.96 * standard_error

    return mean_value - margin, mean_value + margin


# calculate cohens d

def cohens_d(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]

    if len(first) < 2 or len(second) < 2:
        return np.nan

    pooled_variance = (
        (len(first) - 1) * np.var(first, ddof=1)
        + (len(second) - 1) * np.var(second, ddof=1)
    ) / (
        len(first)
        + len(second)
        - 2
    )

    if pooled_variance <= 0:
        return np.nan

    return (
        np.mean(second) - np.mean(first)
    ) / np.sqrt(pooled_variance)


# summarise each bias condition

def summarise_results(results):
    metrics = [
        "final_cooperation",
        "final_noncooperation",
        "remaining_defector_count",
        "remaining_defector_fraction",
        "pd_always_cooperate",
        "pd_conditional",
        "pd_contrarian",
        "pd_always_defect",
    ]

    rows = []

    for (model, bias), group in results.groupby(
        ["model", "initial_cooperation_q"]
    ):
        row = {
            "model": model,
            "initial_cooperation_q": bias,
            "number_simulations": len(group),
        }

        for metric in metrics:
            values = group[
                metric
            ].to_numpy(dtype=np.float64)

            values = values[
                np.isfinite(values)
            ]

            if len(values) == 0:
                row[f"mean_{metric}"] = np.nan
                row[f"sd_{metric}"] = np.nan
                row[f"ci_lower_{metric}"] = np.nan
                row[f"ci_upper_{metric}"] = np.nan
                continue

            lower, upper = get_ci(values)

            row[f"mean_{metric}"] = np.mean(values)
            row[f"sd_{metric}"] = (
                np.std(values, ddof=1)
                if len(values) > 1
                else 0.0
            )
            row[f"ci_lower_{metric}"] = lower
            row[f"ci_upper_{metric}"] = upper

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["model", "initial_cooperation_q"]
        )
        .reset_index(drop=True)
    )


# compare the two bias conditions

def run_stats(results, bias_values):
    if len(bias_values) != 2:
        return pd.DataFrame()

    first_bias = float(bias_values[0])
    second_bias = float(bias_values[1])

    metrics = [
        "final_cooperation",
        "final_noncooperation",
        "remaining_defector_fraction",
        "pd_always_cooperate",
        "pd_conditional",
        "pd_always_defect",
    ]

    rows = []

    for model in ["baseline", "hybrid"]:
        model_results = results[
            results["model"] == model
        ]

        for metric in metrics:
            first = model_results.loc[
                np.isclose(
                    model_results["initial_cooperation_q"],
                    first_bias,
                ),
                metric,
            ].dropna().to_numpy()

            second = model_results.loc[
                np.isclose(
                    model_results["initial_cooperation_q"],
                    second_bias,
                ),
                metric,
            ].dropna().to_numpy()

            if len(first) == 0 or len(second) == 0:
                continue

            welch_t = welch_p = mann_u = mann_p = np.nan

            if (
                SCIPY_AVAILABLE
                and len(first) >= 2
                and len(second) >= 2
            ):
                welch = ttest_ind(
                    first,
                    second,
                    equal_var=False,
                    nan_policy="omit",
                )

                mann = mannwhitneyu(
                    first,
                    second,
                    alternative="two-sided",
                )

                welch_t = welch.statistic
                welch_p = welch.pvalue
                mann_u = mann.statistic
                mann_p = mann.pvalue

            first_mean = np.mean(first)
            second_mean = np.mean(second)

            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "first_bias": first_bias,
                    "second_bias": second_bias,
                    "mean_first": first_mean,
                    "mean_second": second_mean,
                    "mean_change": second_mean - first_mean,
                    "welch_t_statistic": welch_t,
                    "welch_p_value": welch_p,
                    "mann_whitney_u": mann_u,
                    "mann_whitney_p_value": mann_p,
                    "cohens_d": cohens_d(first, second),
                }
            )

    return pd.DataFrame(rows)


# plot final results by initial bias

def plot_bias_results(
    summary,
    metric,
    ylabel,
    title,
    filename,
    output_directory,
    y_limits=None,
):
    bias_values = sorted(
        summary["initial_cooperation_q"].unique()
    )

    x_positions = np.arange(
        len(bias_values)
    )

    width = 0.34

    figure, axis = plt.subplots(figsize=(9, 6))

    for model_index, model in enumerate(
        ["baseline", "hybrid"]
    ):
        model_data = (
            summary[
                summary["model"] == model
            ]
            .set_index("initial_cooperation_q")
            .reindex(bias_values)
        )

        means = model_data[
            f"mean_{metric}"
        ].to_numpy()

        lower = model_data[
            f"ci_lower_{metric}"
        ].to_numpy()

        upper = model_data[
            f"ci_upper_{metric}"
        ].to_numpy()

        offset = (
            model_index - 0.5
        ) * width

        axis.bar(
            x_positions + offset,
            means,
            width=width,
            yerr=np.vstack(
                [means - lower, upper - means]
            ),
            capsize=5,
            color=MODEL_COLOURS[model],
            alpha=0.85,
            label=model.capitalize(),
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [
            f"$Q_C={bias:.2f}$"
            for bias in bias_values
        ]
    )
    axis.set_xlabel("Initial Cooperation Q-Value")
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    if y_limits:
        axis.set_ylim(*y_limits)

    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_directory / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# plot cooperation during training

def plot_learning_curves(
    learning_curves,
    output_directory,
    window,
):
    figure, axis = plt.subplots(figsize=(10, 6))

    for (model, bias), curves in sorted(
        learning_curves.items()
    ):
        if not curves:
            continue

        minimum_length = min(
            len(curve)
            for curve in curves
        )

        matrix = np.vstack(
            [
                curve[:minimum_length]
                for curve in curves
            ]
        )

        mean_curve = np.nanmean(
            matrix,
            axis=0,
        )

        steps = np.arange(
            1,
            minimum_length + 1,
        )

        axis.plot(
            steps,
            moving_average(mean_curve, window),
            color=MODEL_COLOURS[model],
            linestyle=BIAS_LINE_STYLES.get(
                float(bias),
                "-",
            ),
            linewidth=2.6,
            label=(
                f"{model.capitalize()}, "
                f"$Q_C={bias:.2f}$"
            ),
        )

    axis.set_title(
        "Cooperation During Training Under Initial Q-Value Bias"
    )
    axis.set_xlabel("Training Iteration")
    axis.set_ylabel("Cooperation")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "cooperation_learning_curves_initial_bias.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# run initial cooperation bias experiment

def main():
    args = get_args()

    if not 0 < args.final_fraction <= 1:
        raise ValueError(
            "--final-fraction must be between zero and one."
        )

    if not 0 <= args.defector_threshold <= 1:
        raise ValueError(
            "--defector-threshold must be between zero and one."
        )

    args.output = args.output.resolve()
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_files = {
        "baseline": args.baseline_file,
        "hybrid": args.hybrid_file,
    }

    if not args.skip_runs:
        for model, source_file in model_files.items():
            for bias in args.bias_values:
                run_condition(
                    source_file,
                    model,
                    float(bias),
                    args,
                )

    analysis_dir = args.output / "analysis"
    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results, learning_curves = get_results(args)
    summary = summarise_results(results)
    tests = run_stats(
        results,
        args.bias_values,
    )

    results.to_csv(
        analysis_dir
        / "initial_bias_results_by_simulation.csv",
        index=False,
    )

    summary.to_csv(
        analysis_dir
        / "initial_bias_summary.csv",
        index=False,
    )

    tests.to_csv(
        analysis_dir
        / "initial_bias_statistical_tests.csv",
        index=False,
    )

    plot_bias_results(
        summary,
        "final_cooperation",
        "Final Cooperation",
        "Effect of Initial Cooperation Bias on Final Cooperation",
        "final_cooperation_by_initial_bias.png",
        analysis_dir,
        (0, 1),
    )

    plot_bias_results(
        summary,
        "remaining_defector_fraction",
        "Remaining Defector Proportion",
        "Remaining Defectors Under Initial Cooperation Bias",
        "remaining_defectors_by_initial_bias.png",
        analysis_dir,
        (0, 1),
    )

    plot_bias_results(
        summary,
        "final_noncooperation",
        "Shortfall from Complete Cooperation",
        "Remaining Shortfall from 100% Cooperation",
        "cooperation_shortfall_from_100.png",
        analysis_dir,
        (0, 1),
    )

    plot_learning_curves(
        learning_curves,
        analysis_dir,
        args.window,
    )

    print("\nInitial cooperation bias experiment completed.")
    print(f"Results saved in: {analysis_dir}")


if __name__ == "__main__":
    main()