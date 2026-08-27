# analyses the partner-switching learning ablation in the hybrid model
# compares cooperation, switching behaviour, statistical significance and learned policy composition

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


# experiment conditions

CONDITIONS = {
    "hybrid_learned_switching": {
        "label": "Hybrid\nLearned switching",
        "short_label": "Learned switching",
        "learn_switching": True,
    },
    "hybrid_no_switch_learning": {
        "label": "Hybrid\nNo switch learning",
        "short_label": "No switch learning",
        "learn_switching": False,
    },
}

CONDITION_ORDER = [
    "hybrid_learned_switching",
    "hybrid_no_switch_learning",
]

PD_POLICY_NAMES = [
    "Always Cooperate",
    "Tit-for-Tat",
    "Reverse TFT",
    "Always Defect / Tied",
]

SWITCH_POLICY_NAMES = [
    "Always Stay",
    "Stay after C, Switch after D",
    "Switch after C, Stay after D",
    "Always Switch / Tied",
]

COLOURS = {
    "hybrid_learned_switching": "#7b6fd6",
    "hybrid_no_switch_learning": "#c06082",
}


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


# calculate cooperation from game outcomes

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


# calculate mutual cooperation

def get_mutual_cooperation(outcomes):
    outcomes = np.asarray(outcomes, dtype=np.float64)
    total_outcomes = outcomes.sum(axis=1)

    return np.divide(
        outcomes[:, 0],
        total_outcomes,
        out=np.full(outcomes.shape[0], np.nan),
        where=total_outcomes > 0,
    )


# calculate switching rate

def get_switch_rate(switch_counts):
    switch_counts = np.asarray(switch_counts, dtype=np.float64)

    if switch_counts.ndim == 1:
        switch_counts = switch_counts.reshape(1, -1)

    if switch_counts.shape[1] != 2:
        raise ValueError(
            f"Expected two switch columns [stay, switch], "
            f"found {switch_counts.shape}."
        )

    total_decisions = switch_counts.sum(axis=1)

    return np.divide(
        switch_counts[:, 1],
        total_decisions,
        out=np.full(switch_counts.shape[0], np.nan),
        where=total_decisions > 0,
    )


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


# analyse all switch learning runs

def analyse_runs(root_directory, final_window, curve_step):
    run_rows = []

    cooperation_curves = {}
    final_pd_policy_rows = {}
    final_switch_policy_rows = {}

    for condition in CONDITION_ORDER:
        outcome_files = find_results(
            root_directory,
            condition,
            "CountOutcomeT-sim*.txt",
        )
        switch_files = find_results(
            root_directory,
            condition,
            "CountSwitchT-sim*.txt",
        )
        pd_policy_files = find_results(
            root_directory,
            condition,
            "CountPolicyPDTypeT-sim*.txt",
        )
        switch_policy_files = find_results(
            root_directory,
            condition,
            "CountPolicySwTypeT-sim*.txt",
        )

        if not outcome_files:
            raise FileNotFoundError(
                f"No CountOutcomeT files found for {condition}."
            )

        switch_by_sim = {
            get_sim_number(path): path
            for path in switch_files
        }

        pd_policy_by_sim = {
            get_sim_number(path): path
            for path in pd_policy_files
        }

        switch_policy_by_sim = {
            get_sim_number(path): path
            for path in switch_policy_files
        }

        condition_curves = []
        pd_policy_rows = []
        switch_policy_rows = []

        for outcome_file in outcome_files:
            simulation = get_sim_number(outcome_file)

            outcomes = load_data(outcome_file)

            cooperation_curve = get_cooperation(outcomes)
            mutual_cooperation_curve = get_mutual_cooperation(outcomes)

            valid_cooperation = cooperation_curve[
                np.isfinite(cooperation_curve)
            ]

            if valid_cooperation.size == 0:
                raise ValueError(
                    f"No valid cooperation values in {outcome_file}."
                )

            window_size = min(
                final_window,
                valid_cooperation.size,
            )

            final_values = valid_cooperation[-window_size:]

            final_cooperation = np.mean(final_values)
            final_step_cooperation = valid_cooperation[-1]

            final_cooperation_std = (
                np.std(final_values, ddof=1)
                if window_size > 1
                else 0.0
            )

            valid_mutual = mutual_cooperation_curve[
                np.isfinite(mutual_cooperation_curve)
            ]

            mutual_window = min(
                window_size,
                valid_mutual.size,
            )

            final_mutual_cooperation = np.mean(
                valid_mutual[-mutual_window:]
            )

            switch_file = switch_by_sim.get(simulation)

            final_switch_rate = np.nan
            final_step_switch_rate = np.nan

            if switch_file is not None:
                switch_curve = get_switch_rate(
                    load_data(switch_file)
                )

                valid_switch = switch_curve[
                    np.isfinite(switch_curve)
                ]

                if valid_switch.size > 0:
                    switch_window = min(
                        final_window,
                        valid_switch.size,
                    )

                    final_switch_rate = np.mean(
                        valid_switch[-switch_window:]
                    )

                    final_step_switch_rate = valid_switch[-1]

            condition_info = CONDITIONS[condition]

            run_rows.append(
                {
                    "condition": condition,
                    "label": condition_info["short_label"],
                    "learn_switching": condition_info["learn_switching"],
                    "simulation": simulation,
                    "iterations": len(cooperation_curve),
                    "final_window_size": window_size,
                    "final_window_cooperation": final_cooperation,
                    "final_step_cooperation": final_step_cooperation,
                    "within_run_final_window_std": final_cooperation_std,
                    "final_window_mutual_cooperation":
                        final_mutual_cooperation,
                    "final_window_switch_rate": final_switch_rate,
                    "final_step_switch_rate": final_step_switch_rate,
                    "outcome_file": outcome_file,
                    "switch_file": switch_file,
                }
            )

            condition_curves.append(
                cooperation_curve[::curve_step]
            )

            pd_policy_file = pd_policy_by_sim.get(simulation)

            if pd_policy_file is not None:
                pd_policy_rows.append(
                    load_data(pd_policy_file)[-1].astype(
                        np.float64
                    )
                )

            switch_policy_file = switch_policy_by_sim.get(
                simulation
            )

            if switch_policy_file is not None:
                switch_policy_rows.append(
                    load_data(switch_policy_file)[-1].astype(
                        np.float64
                    )
                )

        minimum_length = min(
            len(curve)
            for curve in condition_curves
        )

        cooperation_curves[condition] = np.vstack(
            [
                curve[:minimum_length]
                for curve in condition_curves
            ]
        )

        if pd_policy_rows:
            final_pd_policy_rows[condition] = np.vstack(
                pd_policy_rows
            )

        if switch_policy_rows:
            final_switch_policy_rows[condition] = np.vstack(
                switch_policy_rows
            )

    return (
        pd.DataFrame(run_rows),
        cooperation_curves,
        final_pd_policy_rows,
        final_switch_policy_rows,
    )


# summarise each condition

def summarise_conditions(run_data):
    rows = []

    for condition in CONDITION_ORDER:
        condition_data = run_data[
            run_data["condition"] == condition
        ]

        cooperation_values = condition_data[
            "final_window_cooperation"
        ].to_numpy(dtype=np.float64)

        switch_values = condition_data[
            "final_window_switch_rate"
        ].to_numpy(dtype=np.float64)

        (
            cooperation_mean,
            cooperation_ci_low,
            cooperation_ci_high,
        ) = get_ci(cooperation_values)

        (
            switch_mean,
            switch_ci_low,
            switch_ci_high,
        ) = get_ci(switch_values)

        cooperation_std = (
            np.std(cooperation_values, ddof=1)
            if len(cooperation_values) > 1
            else np.nan
        )

        switch_std = (
            np.std(switch_values, ddof=1)
            if len(switch_values) > 1
            else np.nan
        )

        cooperation_sem = (
            cooperation_std
            / np.sqrt(len(cooperation_values))
            if len(cooperation_values) > 1
            else np.nan
        )

        rows.append(
            {
                "condition": condition,
                "label": CONDITIONS[condition]["short_label"],
                "learn_switching": CONDITIONS[condition]["learn_switching"],
                "n_simulations": len(cooperation_values),
                "mean_final_cooperation": cooperation_mean,
                "std_final_cooperation": cooperation_std,
                "sem_final_cooperation": cooperation_sem,
                "ci95_lower_cooperation": cooperation_ci_low,
                "ci95_upper_cooperation": cooperation_ci_high,
                "minimum_cooperation": np.min(cooperation_values),
                "maximum_cooperation": np.max(cooperation_values),
                "median_cooperation": np.median(cooperation_values),
                "mean_final_switch_rate": switch_mean,
                "std_final_switch_rate": switch_std,
                "ci95_lower_switch_rate": switch_ci_low,
                "ci95_upper_switch_rate": switch_ci_high,
            }
        )

    return pd.DataFrame(rows)


# compare learned and fixed switching

def run_statistical_tests(run_data):
    learned_cooperation = run_data.loc[
        run_data["condition"] == "hybrid_learned_switching",
        "final_window_cooperation",
    ].to_numpy(dtype=np.float64)

    fixed_cooperation = run_data.loc[
        run_data["condition"] == "hybrid_no_switch_learning",
        "final_window_cooperation",
    ].to_numpy(dtype=np.float64)

    learned_switching = run_data.loc[
        run_data["condition"] == "hybrid_learned_switching",
        "final_window_switch_rate",
    ].to_numpy(dtype=np.float64)

    fixed_switching = run_data.loc[
        run_data["condition"] == "hybrid_no_switch_learning",
        "final_window_switch_rate",
    ].to_numpy(dtype=np.float64)

    cooperation_welch = stats.ttest_ind(
        learned_cooperation,
        fixed_cooperation,
        equal_var=False,
        nan_policy="omit",
    )

    cooperation_mann = stats.mannwhitneyu(
        learned_cooperation,
        fixed_cooperation,
        alternative="two-sided",
    )

    cooperation_effect = cohens_d(
        learned_cooperation,
        fixed_cooperation,
    )

    switching_welch = stats.ttest_ind(
        learned_switching,
        fixed_switching,
        equal_var=False,
        nan_policy="omit",
    )

    switching_mann = stats.mannwhitneyu(
        learned_switching,
        fixed_switching,
        alternative="two-sided",
    )

    switching_effect = cohens_d(
        learned_switching,
        fixed_switching,
    )

    mean_learned = np.mean(learned_cooperation)
    mean_fixed = np.mean(fixed_cooperation)

    cooperation_difference = (
        mean_learned - mean_fixed
    )

    proportional_difference = (
        cooperation_difference / mean_learned
        if mean_learned != 0
        else np.nan
    )

    rows = [
        {
            "metric": "final_cooperation",
            "comparison": "Learned switching vs no switch learning",
            "mean_learned_switching": mean_learned,
            "mean_no_switch_learning": mean_fixed,
            "absolute_difference": cooperation_difference,
            "proportional_difference": proportional_difference,
            "welch_t_statistic": cooperation_welch.statistic,
            "welch_p_value": cooperation_welch.pvalue,
            "mann_whitney_u": cooperation_mann.statistic,
            "mann_whitney_p_value": cooperation_mann.pvalue,
            "cohens_d": cooperation_effect,
            "effect_size_interpretation":
                get_effect_label(cooperation_effect),
        },
        {
            "metric": "final_switch_rate",
            "comparison": "Learned switching vs no switch learning",
            "mean_learned_switching": np.mean(learned_switching),
            "mean_no_switch_learning": np.mean(fixed_switching),
            "absolute_difference":
                np.mean(learned_switching)
                - np.mean(fixed_switching),
            "proportional_difference": np.nan,
            "welch_t_statistic": switching_welch.statistic,
            "welch_p_value": switching_welch.pvalue,
            "mann_whitney_u": switching_mann.statistic,
            "mann_whitney_p_value": switching_mann.pvalue,
            "cohens_d": switching_effect,
            "effect_size_interpretation":
                get_effect_label(switching_effect),
        },
    ]

    return pd.DataFrame(rows)


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
            mean_value - row["ci95_lower_cooperation"]
        )
        upper_errors.append(
            row["ci95_upper_cooperation"] - mean_value
        )

    figure, axis = plt.subplots(figsize=(9, 7))

    bars = axis.bar(
        x_positions,
        means,
        yerr=np.vstack([lower_errors, upper_errors]),
        capsize=8,
        color=[
            COLOURS[condition]
            for condition in CONDITION_ORDER
        ],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
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
            0.035,
            len(values),
        )

        axis.scatter(
            np.full(len(values), x_position) + jitter,
            values,
            color="black",
            s=32,
            alpha=0.75,
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
    axis.set_xlabel("Switch-learning condition")
    axis.set_title("Partner-Switching Learning Ablation")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, mean_value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(mean_value + 0.05, 1.02),
            f"{mean_value:.3f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            output_directory,
            "switch_learning_final_cooperation.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# plot cooperation learning curves

def plot_learning_curves(
    cooperation_curves,
    curve_step,
    output_directory,
):
    line_styles = {
        "hybrid_learned_switching": "-",
        "hybrid_no_switch_learning": "--",
    }

    figure, axis = plt.subplots(figsize=(11, 7))

    for condition in CONDITION_ORDER:
        curves = cooperation_curves[condition]
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

        x_values = (
            np.arange(len(mean_curve))
            * curve_step
        )

        axis.plot(
            x_values,
            mean_curve,
            label=CONDITIONS[condition]["short_label"],
            color=COLOURS[condition],
            linestyle=line_styles[condition],
            linewidth=2.4,
        )

        axis.fill_between(
            x_values,
            mean_curve - confidence_margin,
            mean_curve + confidence_margin,
            color=COLOURS[condition],
            alpha=0.15,
        )

    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Mean cooperation rate")
    axis.set_title(
        "Cooperation Learning Curves: Switch-Learning Ablation"
    )
    axis.set_ylim(0.0, 1.05)
    axis.grid(linestyle="--", alpha=0.3)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            output_directory,
            "switch_learning_learning_curves.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# plot final switching rate

def plot_final_switch_rate(
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

        mean_value = row["mean_final_switch_rate"]

        means.append(mean_value)
        lower_errors.append(
            mean_value - row["ci95_lower_switch_rate"]
        )
        upper_errors.append(
            row["ci95_upper_switch_rate"] - mean_value
        )

    figure, axis = plt.subplots(figsize=(9, 7))

    axis.bar(
        x_positions,
        means,
        yerr=np.vstack([lower_errors, upper_errors]),
        capsize=8,
        color=[
            COLOURS[condition]
            for condition in CONDITION_ORDER
        ],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    random_generator = np.random.default_rng(4321)

    for x_position, condition in zip(
        x_positions,
        CONDITION_ORDER,
    ):
        values = run_data.loc[
            run_data["condition"] == condition,
            "final_window_switch_rate",
        ].to_numpy(dtype=np.float64)

        jitter = random_generator.normal(
            0.0,
            0.035,
            len(values),
        )

        axis.scatter(
            np.full(len(values), x_position) + jitter,
            values,
            color="black",
            s=32,
            alpha=0.75,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [
            CONDITIONS[condition]["label"]
            for condition in CONDITION_ORDER
        ]
    )
    axis.set_ylabel(
        "Mean switching rate\n(last training window)"
    )
    axis.set_xlabel("Switch-learning condition")
    axis.set_title("Final Switching Behaviour")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", linestyle="--", alpha=0.35)

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            output_directory,
            "switch_learning_final_switch_rate.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# plot final policy composition

def plot_policy_composition(
    policy_rows,
    policy_names,
    title,
    output_filename,
    output_directory,
):
    available_conditions = [
        condition
        for condition in CONDITION_ORDER
        if condition in policy_rows
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

    policy_colours = [
        "#c06082",
        "#db8caf",
        "#9a8be0",
        "#7b6fd6",
    ]

    figure, axis = plt.subplots(figsize=(11, 7))

    for policy_index, policy_name in enumerate(
        policy_names
    ):
        proportions = []

        for condition in available_conditions:
            policy_matrix = policy_rows[condition]

            mean_count = np.mean(
                policy_matrix[:, policy_index]
            )

            mean_total = np.mean(
                policy_matrix.sum(axis=1)
            )

            proportions.append(
                mean_count / mean_total
                if mean_total > 0
                else np.nan
            )

        axis.bar(
            x_positions,
            proportions,
            bottom=bottoms,
            label=policy_name,
            color=policy_colours[policy_index],
            edgecolor="white",
            linewidth=0.7,
        )

        bottoms += np.asarray(proportions)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [
            CONDITIONS[condition]["label"]
            for condition in available_conditions
        ]
    )
    axis.set_ylabel("Mean proportion of agents")
    axis.set_xlabel("Switch-learning condition")
    axis.set_title(title)
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
            output_filename,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# run switch learning analysis

def main():
    parser = argparse.ArgumentParser(
        description="Analyse the switch-learning ablation."
    )

    parser.add_argument(
        "--root",
        default="switch_learning_ablation_results",
        help="Root directory containing the ablation results.",
    )
    parser.add_argument(
        "--final-window",
        type=int,
        default=10000,
        help="Final iterations used for stable behaviour.",
    )
    parser.add_argument(
        "--curve-step",
        type=int,
        default=1000,
        help="Sampling interval used for learning curves.",
    )
    parser.add_argument(
        "--output",
        default="switch_learning_ablation_analysis",
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
        cooperation_curves,
        final_pd_policy_rows,
        final_switch_policy_rows,
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
            "switch_learning_run_summary.csv",
        ),
        index=False,
    )

    condition_summary.to_csv(
        os.path.join(
            args.output,
            "switch_learning_condition_summary.csv",
        ),
        index=False,
    )

    statistical_tests.to_csv(
        os.path.join(
            args.output,
            "switch_learning_statistical_tests.csv",
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
        cooperation_curves,
        args.curve_step,
        args.output,
    )

    plot_final_switch_rate(
        run_data,
        condition_summary,
        args.output,
    )

    plot_policy_composition(
        final_pd_policy_rows,
        PD_POLICY_NAMES,
        "Final Learned Prisoner's Dilemma Policy Composition",
        "switch_learning_pd_policy_composition.png",
        args.output,
    )

    plot_policy_composition(
        final_switch_policy_rows,
        SWITCH_POLICY_NAMES,
        "Final Learned Partner-Switching Policy Composition",
        "switch_learning_switch_policy_composition.png",
        args.output,
    )

    print("\nSwitch-learning ablation analysis completed.")
    print(f"Results saved in: {args.output}")


if __name__ == "__main__":
    main()