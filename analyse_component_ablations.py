# analyses the discounting component ablation for the baseline and hybrid models
# compares cooperation, statistical significance, learning curves and final policy composition

import argparse
import glob
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# experimental conditions

CONDITIONS = {
    "baseline_full": {
        "model": "Baseline",
        "gamma": 1.0,
        "label": "Baseline\nFull",
    },
    "baseline_no_discounting": {
        "model": "Baseline",
        "gamma": 0.0,
        "label": "Baseline\nNo discounting",
    },
    "hybrid_full": {
        "model": "Hybrid",
        "gamma": 1.0,
        "label": "Hybrid\nFull",
    },
    "hybrid_no_discounting": {
        "model": "Hybrid",
        "gamma": 0.0,
        "label": "Hybrid\nNo discounting",
    },
}

CONDITION_ORDER = [
    "baseline_full",
    "baseline_no_discounting",
    "hybrid_full",
    "hybrid_no_discounting",
]

POLICY_NAMES = [
    "Always Cooperate",
    "Tit-for-Tat",
    "Reverse TFT",
    "Always Defect / Tied",
]

CONDITION_COLOURS = [
    "#c06082",
    "#e2a6bb",
    "#7b6fd6",
    "#afa7e8",
]

POLICY_COLOURS = [
    "#c06082",
    "#db8caf",
    "#9a8be0",
    "#7b6fd6",
]


# calculate cooperation from outcome counts

def get_cooperation(outcomes):
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if outcomes.ndim == 1:
        outcomes = outcomes.reshape(1, -1)

    if outcomes.shape[1] != 4:
        raise ValueError(
            f"Expected four outcome columns [CC, CD, DC, DD], "
            f"found {outcomes.shape}."
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


# get simulation number from filename

def get_sim_number(file_path):
    stem = Path(file_path).stem

    if "sim" not in stem:
        return -1

    try:
        return int(stem.split("sim")[-1])
    except ValueError:
        return -1


# find result files for one condition

def find_results(root_directory, condition, filename_pattern):
    pattern = os.path.join(
        root_directory,
        condition,
        "**",
        filename_pattern,
    )

    return sorted(
        glob.glob(pattern, recursive=True),
        key=get_sim_number,
    )


# load simulation data

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=",")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data


# calculate confidence interval

def get_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean_value = np.mean(values)

    if len(values) == 1:
        return mean_value, np.nan, np.nan

    standard_error = stats.sem(values)
    critical_value = stats.t.ppf(
        (1.0 + confidence) / 2.0,
        df=len(values) - 1,
    )

    margin = critical_value * standard_error

    return (
        mean_value,
        mean_value - margin,
        mean_value + margin,
    )


# calculate cohens d

def cohens_d(first_group, second_group):
    first_group = np.asarray(first_group, dtype=np.float64)
    second_group = np.asarray(second_group, dtype=np.float64)

    first_group = first_group[np.isfinite(first_group)]
    second_group = second_group[np.isfinite(second_group)]

    n_first = len(first_group)
    n_second = len(second_group)

    if n_first < 2 or n_second < 2:
        return np.nan

    first_variance = np.var(first_group, ddof=1)
    second_variance = np.var(second_group, ddof=1)

    pooled_variance = (
        (n_first - 1) * first_variance
        + (n_second - 1) * second_variance
    ) / (n_first + n_second - 2)

    if pooled_variance <= 0:
        if np.isclose(
            np.mean(first_group),
            np.mean(second_group),
        ):
            return 0.0

        return np.inf

    return (
        np.mean(first_group)
        - np.mean(second_group)
    ) / np.sqrt(pooled_variance)


# label effect size

def get_effect_label(effect_size):
    if not np.isfinite(effect_size):
        return "undefined"

    effect_size = abs(effect_size)

    if effect_size < 0.2:
        return "negligible"
    if effect_size < 0.5:
        return "small"
    if effect_size < 0.8:
        return "medium"

    return "large"


# analyse all component ablation runs

def analyse_runs(root_directory, final_window, curve_step):
    run_rows = []
    downsampled_curves = {}
    final_policy_rows = {}

    for condition in CONDITION_ORDER:
        outcome_files = find_results(
            root_directory,
            condition,
            "CountOutcomeT-sim*.txt",
        )

        policy_files = find_results(
            root_directory,
            condition,
            "CountPolicyPDTypeT-sim*.txt",
        )

        if not outcome_files:
            raise FileNotFoundError(
                f"No CountOutcomeT files found for {condition}."
            )

        policy_by_sim = {
            get_sim_number(path): path
            for path in policy_files
        }

        condition_curves = []
        condition_policy_rows = []

        for outcome_file in outcome_files:
            simulation = get_sim_number(outcome_file)

            outcomes = load_data(outcome_file)
            cooperation_curve = get_cooperation(outcomes)

            valid_curve = cooperation_curve[
                np.isfinite(cooperation_curve)
            ]

            if valid_curve.size == 0:
                raise ValueError(
                    f"No valid outcome rows in {outcome_file}"
                )

            window_size = min(
                final_window,
                valid_curve.size,
            )

            final_values = valid_curve[-window_size:]

            final_cooperation = np.mean(final_values)
            final_step_cooperation = valid_curve[-1]

            final_window_std = (
                np.std(final_values, ddof=1)
                if window_size > 1
                else 0.0
            )

            final_mutual_cooperation = np.mean(
                outcomes[-window_size:, 0]
                / np.maximum(
                    outcomes[-window_size:].sum(axis=1),
                    1,
                )
            )

            condition_info = CONDITIONS[condition]

            run_rows.append(
                {
                    "condition": condition,
                    "model": condition_info["model"],
                    "gamma": condition_info["gamma"],
                    "simulation": simulation,
                    "iterations": len(cooperation_curve),
                    "final_window_size": window_size,
                    "final_window_cooperation": final_cooperation,
                    "final_step_cooperation": final_step_cooperation,
                    "within_run_final_window_std": final_window_std,
                    "final_window_mutual_cooperation_rate":
                        final_mutual_cooperation,
                    "outcome_file": outcome_file,
                }
            )

            condition_curves.append(
                cooperation_curve[::curve_step]
            )

            policy_file = policy_by_sim.get(simulation)

            if policy_file is not None:
                condition_policy_rows.append(
                    load_data(policy_file)[-1].astype(
                        np.float64
                    )
                )

        minimum_length = min(
            len(curve)
            for curve in condition_curves
        )

        downsampled_curves[condition] = np.vstack(
            [
                curve[:minimum_length]
                for curve in condition_curves
            ]
        )

        if condition_policy_rows:
            final_policy_rows[condition] = np.vstack(
                condition_policy_rows
            )

    return (
        pd.DataFrame(run_rows),
        downsampled_curves,
        final_policy_rows,
    )


# summarise each condition

def summarise_conditions(run_data):
    rows = []

    for condition in CONDITION_ORDER:
        values = run_data.loc[
            run_data["condition"] == condition,
            "final_window_cooperation",
        ].to_numpy(dtype=np.float64)

        mean_value, ci_low, ci_high = get_ci(values)

        standard_deviation = (
            np.std(values, ddof=1)
            if len(values) > 1
            else np.nan
        )

        standard_error = (
            standard_deviation / np.sqrt(len(values))
            if len(values) > 1
            else np.nan
        )

        rows.append(
            {
                "condition": condition,
                "model": CONDITIONS[condition]["model"],
                "gamma": CONDITIONS[condition]["gamma"],
                "n_simulations": len(values),
                "mean_final_cooperation": mean_value,
                "std_final_cooperation": standard_deviation,
                "sem_final_cooperation": standard_error,
                "ci95_lower": ci_low,
                "ci95_upper": ci_high,
                "minimum": np.min(values),
                "maximum": np.max(values),
                "median": np.median(values),
            }
        )

    return pd.DataFrame(rows)


# compare two ablation conditions

def compare_conditions(
    run_data,
    full_condition,
    ablated_condition,
    comparison_name,
):
    full_values = run_data.loc[
        run_data["condition"] == full_condition,
        "final_window_cooperation",
    ].to_numpy(dtype=np.float64)

    ablated_values = run_data.loc[
        run_data["condition"] == ablated_condition,
        "final_window_cooperation",
    ].to_numpy(dtype=np.float64)

    welch_test = stats.ttest_ind(
        full_values,
        ablated_values,
        equal_var=False,
        nan_policy="omit",
    )

    mann_whitney = stats.mannwhitneyu(
        full_values,
        ablated_values,
        alternative="two-sided",
    )

    effect_size = cohens_d(
        full_values,
        ablated_values,
    )

    mean_full = np.mean(full_values)
    mean_ablated = np.mean(ablated_values)

    absolute_drop = mean_full - mean_ablated

    proportional_drop = (
        absolute_drop / mean_full
        if mean_full != 0
        else np.nan
    )

    return {
        "comparison": comparison_name,
        "full_condition": full_condition,
        "ablated_condition": ablated_condition,
        "n_full": len(full_values),
        "n_ablated": len(ablated_values),
        "mean_full": mean_full,
        "mean_ablated": mean_ablated,
        "absolute_cooperation_drop": absolute_drop,
        "proportional_drop": proportional_drop,
        "welch_t_statistic": welch_test.statistic,
        "welch_p_value": welch_test.pvalue,
        "mann_whitney_u": mann_whitney.statistic,
        "mann_whitney_p_value": mann_whitney.pvalue,
        "cohens_d": effect_size,
        "effect_size_interpretation":
            get_effect_label(effect_size),
    }


# run statistical comparisons

def run_statistical_tests(run_data):
    comparisons = [
        compare_conditions(
            run_data,
            "baseline_full",
            "baseline_no_discounting",
            "Baseline: full vs no discounting",
        ),
        compare_conditions(
            run_data,
            "hybrid_full",
            "hybrid_no_discounting",
            "Hybrid: full vs no discounting",
        ),
        compare_conditions(
            run_data,
            "hybrid_full",
            "baseline_full",
            "Full models: hybrid vs baseline",
        ),
        compare_conditions(
            run_data,
            "hybrid_no_discounting",
            "baseline_no_discounting",
            "No discounting: hybrid vs baseline",
        ),
    ]

    return pd.DataFrame(comparisons)


# plot final cooperation

def plot_final_cooperation(
    run_data,
    condition_summary,
    output_directory,
):
    x_positions = np.arange(len(CONDITION_ORDER))

    means = []
    lower_errors = []
    upper_errors = []

    for condition in CONDITION_ORDER:
        row = condition_summary[
            condition_summary["condition"] == condition
        ].iloc[0]

        mean_value = row["mean_final_cooperation"]

        means.append(mean_value)
        lower_errors.append(
            mean_value - row["ci95_lower"]
        )
        upper_errors.append(
            row["ci95_upper"] - mean_value
        )

    figure, axis = plt.subplots(figsize=(11, 7))

    bars = axis.bar(
        x_positions,
        means,
        yerr=np.vstack([lower_errors, upper_errors]),
        capsize=7,
        color=CONDITION_COLOURS,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.88,
        zorder=2,
    )

    random_generator = np.random.default_rng(1234)

    for x_position, condition in zip(
        x_positions,
        CONDITION_ORDER,
    ):
        values = run_data.loc[
            run_data["condition"] == condition,
            "final_window_cooperation",
        ].to_numpy(dtype=np.float64)

        jitter = random_generator.normal(
            0.0,
            0.045,
            len(values),
        )

        axis.scatter(
            np.full(len(values), x_position) + jitter,
            values,
            color="black",
            s=28,
            alpha=0.75,
            zorder=3,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [
            CONDITIONS[condition]["label"]
            for condition in CONDITION_ORDER
        ]
    )
    axis.set_ylabel(
        "Mean cooperation rate\n(last training window)"
    )
    axis.set_xlabel("Experimental condition")
    axis.set_title("Discounting Component Ablation Study")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, mean_value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(mean_value + 0.055, 1.02),
            f"{mean_value:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            output_directory,
            "component_ablation_final_cooperation.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# plot cooperation learning curves

def plot_learning_curves(
    downsampled_curves,
    curve_step,
    output_directory,
):
    colours = {
        "baseline_full": "#c06082",
        "baseline_no_discounting": "#e2a6bb",
        "hybrid_full": "#7b6fd6",
        "hybrid_no_discounting": "#afa7e8",
    }

    line_styles = {
        "baseline_full": "-",
        "baseline_no_discounting": "--",
        "hybrid_full": "-",
        "hybrid_no_discounting": "--",
    }

    labels = {
        "baseline_full": "Baseline full",
        "baseline_no_discounting": "Baseline without discounting",
        "hybrid_full": "Hybrid full",
        "hybrid_no_discounting": "Hybrid without discounting",
    }

    figure, axis = plt.subplots(figsize=(11, 7))

    for condition in CONDITION_ORDER:
        curves = downsampled_curves[condition]
        mean_curve = np.nanmean(curves, axis=0)

        if curves.shape[0] > 1:
            standard_error = stats.sem(
                curves,
                axis=0,
                nan_policy="omit",
            )

            critical_value = stats.t.ppf(
                0.975,
                df=curves.shape[0] - 1,
            )

            confidence_margin = (
                critical_value * standard_error
            )
        else:
            confidence_margin = np.zeros_like(
                mean_curve
            )

        x_values = np.arange(
            len(mean_curve)
        ) * curve_step

        axis.plot(
            x_values,
            mean_curve,
            label=labels[condition],
            color=colours[condition],
            linestyle=line_styles[condition],
            linewidth=2.2,
        )

        axis.fill_between(
            x_values,
            mean_curve - confidence_margin,
            mean_curve + confidence_margin,
            color=colours[condition],
            alpha=0.14,
        )

    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Mean cooperation rate")
    axis.set_title(
        "Cooperation Learning Curves Across Ablation Conditions"
    )
    axis.set_ylim(0.0, 1.05)
    axis.grid(linestyle="--", alpha=0.3)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            output_directory,
            "component_ablation_learning_curves.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# plot final policy composition

def plot_policy_composition(
    final_policy_rows,
    output_directory,
):
    available_conditions = [
        condition
        for condition in CONDITION_ORDER
        if condition in final_policy_rows
    ]

    if not available_conditions:
        return

    x_positions = np.arange(
        len(available_conditions)
    )

    bottoms = np.zeros(
        len(available_conditions),
        dtype=np.float64,
    )

    figure, axis = plt.subplots(figsize=(12, 7))

    for policy_index, policy_name in enumerate(
        POLICY_NAMES
    ):
        values = []

        for condition in available_conditions:
            policy_matrix = final_policy_rows[condition]

            mean_count = np.mean(
                policy_matrix[:, policy_index]
            )

            total_agents = np.mean(
                policy_matrix.sum(axis=1)
            )

            values.append(
                mean_count / total_agents
                if total_agents > 0
                else np.nan
            )

        axis.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=policy_name,
            color=POLICY_COLOURS[policy_index],
            edgecolor="white",
            linewidth=0.7,
        )

        bottoms += np.asarray(values)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [
            CONDITIONS[condition]["label"]
            for condition in available_conditions
        ]
    )
    axis.set_ylabel("Mean proportion of agents")
    axis.set_xlabel("Experimental condition")
    axis.set_title("Final Learned PD Policy Composition")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend(
        title="Policy type",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            output_directory,
            "component_ablation_policy_composition.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# run component ablation analysis

def main():
    parser = argparse.ArgumentParser(
        description="Analyse the component ablation study."
    )

    parser.add_argument(
        "--root",
        default="component_ablation_results",
        help="Root directory containing the ablation conditions.",
    )
    parser.add_argument(
        "--final-window",
        type=int,
        default=10000,
        help="Final training iterations used for cooperation.",
    )
    parser.add_argument(
        "--curve-step",
        type=int,
        default=1000,
        help="Sampling interval used for learning curves.",
    )
    parser.add_argument(
        "--output",
        default="component_ablation_analysis",
        help="Directory used for analysis outputs.",
    )

    args = parser.parse_args()

    if args.final_window <= 0:
        raise ValueError(
            "--final-window must be greater than zero."
        )

    if args.curve_step <= 0:
        raise ValueError(
            "--curve-step must be greater than zero."
        )

    os.makedirs(args.output, exist_ok=True)

    (
        run_data,
        downsampled_curves,
        final_policy_rows,
    ) = analyse_runs(
        args.root,
        args.final_window,
        args.curve_step,
    )

    condition_summary = summarise_conditions(
        run_data
    )

    statistical_tests = run_statistical_tests(
        run_data
    )

    run_data.to_csv(
        os.path.join(
            args.output,
            "component_ablation_run_summary.csv",
        ),
        index=False,
    )

    condition_summary.to_csv(
        os.path.join(
            args.output,
            "component_ablation_condition_summary.csv",
        ),
        index=False,
    )

    statistical_tests.to_csv(
        os.path.join(
            args.output,
            "component_ablation_statistical_tests.csv",
        ),
        index=False,
    )

    # create figures

    plot_final_cooperation(
        run_data,
        condition_summary,
        args.output,
    )

    plot_learning_curves(
        downsampled_curves,
        args.curve_step,
        args.output,
    )

    plot_policy_composition(
        final_policy_rows,
        args.output,
    )

    print("\nComponent ablation analysis completed.")
    print(f"Results saved in: {args.output}")


if __name__ == "__main__":
    main()