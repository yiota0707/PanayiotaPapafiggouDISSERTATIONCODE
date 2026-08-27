# runs and analyses the cooperation-bonus experiment for baseline and hybrid models
# tests how different cooperation rewards affect cooperation, residual defection and learned strategies

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
LIGHT_PURPLE = "#b9b2ee"
LIGHT_PINK = "#e6a8bd"

MODEL_COLOURS = {
    "baseline": PURPLE,
    "hybrid": PINK,
}

MODEL_MARKERS = {
    "baseline": "o",
    "hybrid": "s",
}

BETA_LINE_STYLES = {
    0.00: "--",
    0.10: ":",
    0.25: "-.",
    0.50: "-",
    1.00: (0, (5, 1)),
}

STRATEGY_NAMES = [
    "Always Cooperate",
    "Conditional",
    "Contrarian",
    "Always Defect",
]

STRATEGY_COLOURS = [
    PINK,
    PURPLE,
    LIGHT_PINK,
    LIGHT_PURPLE,
]

SIMULATION_PATTERN = re.compile(
    r"CountOutcomeT-sim(?P<simulation>\d+)\.txt$"
)


# command line arguments

def get_args():
    parser = argparse.ArgumentParser(
        description="Run and analyse the cooperation-bonus ablation."
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
        default=Path("cooperation_bonus_ablation_results"),
    )
    parser.add_argument(
        "--beta-values",
        nargs="+",
        type=float,
        default=[0.00, 0.10, 0.25, 0.50, 1.00],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "hybrid"],
        default=["baseline", "hybrid"],
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


# replace one section of the model code

def replace_code(source, old, new, description):
    if old not in source:
        raise RuntimeError(
            f"Could not modify {description}. Expected code was not found."
        )

    return source.replace(old, new, 1)


# create temporary model with cooperation bonus

def make_bonus_model(source_path, temporary_path):
    source_path = source_path.resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Model file not found: {source_path}")

    source = source_path.read_text(encoding="utf-8")

    # add cooperation bonus to the q update
    old_update_function = """def update_q_values(
    Q,
    memory_states,
    memory_actions,
    memory_rewards,
    memory_lengths,
    learning_rate,
    gamma,
):"""

    new_update_function = """def update_q_values(
    Q,
    memory_states,
    memory_actions,
    memory_rewards,
    memory_lengths,
    learning_rate,
    gamma,
    coop_bonus,
):"""

    source = replace_code(
        source,
        old_update_function,
        new_update_function,
        "update_q_values function",
    )

    old_update = """            Q[agent, state_index, action] = (
                (1.0 - learning_rate) * Q[agent, state_index, action]
                + learning_rate * running_return
            )"""

    new_update = """            target = running_return

            if memory_states[agent, memory_index] >= 100 and action == 0:
                target += coop_bonus

            Q[agent, state_index, action] = (
                (1.0 - learning_rate) * Q[agent, state_index, action]
                + learning_rate * target
            )"""

    source = replace_code(
        source,
        old_update,
        new_update,
        "Q-learning update",
    )

    # pass the bonus through the simulation
    old_simulation_function = """def run_simulation(
    number_agents,
    rounds_per_game,
    training_horizon,
    learning_rate,
    m,
    tau,
    seed,
):"""

    new_simulation_function = """def run_simulation(
    number_agents,
    rounds_per_game,
    training_horizon,
    learning_rate,
    m,
    tau,
    seed,
    coop_bonus,
):"""

    source = replace_code(
        source,
        old_simulation_function,
        new_simulation_function,
        "run_simulation function",
    )

    old_update_call = """        update_q_values(
            Q,
            memory_states,
            memory_actions,
            memory_rewards,
            memory_lengths,
            learning_rate,
            gamma,
        )"""

    new_update_call = """        update_q_values(
            Q,
            memory_states,
            memory_actions,
            memory_rewards,
            memory_lengths,
            learning_rate,
            gamma,
            coop_bonus,
        )"""

    source = replace_code(
        source,
        old_update_call,
        new_update_call,
        "update_q_values call",
    )

    # add the bonus argument to the batch runner
    old_batch = """def run_experiment_batch(
    gameName,
    Nact,
    roundsG,
    algoName,
    lr,
    Nagent,
    simStart,
    Nsim,
    tStart,
    T,
    m,
    seed=None,
    tau=1.0,
):"""

    new_batch = """def run_experiment_batch(
    gameName,
    Nact,
    roundsG,
    algoName,
    lr,
    Nagent,
    simStart,
    Nsim,
    tStart,
    T,
    m,
    seed=None,
    tau=1.0,
    coop_bonus=0.0,
):"""

    source = replace_code(
        source,
        old_batch,
        new_batch,
        "run_experiment_batch function",
    )

    old_simulation_call = """        output = run_simulation(
            Nagent,
            roundsG,
            T,
            lr,
            m,
            tau,
            simulation_seed,
        )"""

    new_simulation_call = """        output = run_simulation(
            Nagent,
            roundsG,
            T,
            lr,
            m,
            tau,
            simulation_seed,
            coop_bonus,
        )"""

    source = replace_code(
        source,
        old_simulation_call,
        new_simulation_call,
        "run_simulation call",
    )

    # read beta as the final command-line argument
    source = replace_code(
        source,
        "    tau = 1.0\n",
        "    tau = 1.0\n    coop_bonus = 0.0\n",
        "cooperation bonus default",
    )

    source = replace_code(
        source,
        "    if len(sys.argv) > 5:\n        training_horizon = int(sys.argv[5])",
        (
            "    if len(sys.argv) > 5:\n"
            "        T = int(sys.argv[5])\n"
            "    if len(sys.argv) > 6:\n"
            "        coop_bonus = float(sys.argv[6])"
        ),
        "cooperation bonus argument",
    )

    old_call = """    run_experiment_batch(
        gameName="PD2",
        Nact=2,
        roundsG=rounds_per_game,
        algoName="{algorithm}",
        lr=learning_rate,
        Nagent=number_agents,
        simStart=1,
        Nsim=number_simulations,
        tStart=0,
        T=training_horizon,
        m=assortativity,
        seed=1234,
        tau=tau,
    )"""

    algorithm = (
        "Qpast1-b-baseline"
        if "baseline" in source_path.name
        else "Qpast1-b-hybrid"
    )

    old_call = old_call.format(algorithm=algorithm)

    new_call = old_call.replace(
        "        tau=tau,\n    )",
        "        tau=tau,\n        coop_bonus=coop_bonus,\n    )",
    )

    source = replace_code(
        source,
        old_call,
        new_call,
        "run_experiment_batch call",
    )

    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(source, encoding="utf-8")

# get output directory for one condition

def get_condition_dir(output_root, model, beta):
    return output_root / model / f"beta{beta:.2f}"


# count completed simulations

def count_sims(directory):
    if not directory.exists():
        return 0

    return len(list(directory.rglob("CountOutcomeT-sim*.txt")))


# run one model and beta condition

def run_bonus_condition(source_file, model, beta, args):
    condition = get_condition_dir(args.output, model, beta)
    completed = count_sims(condition)

    if completed >= args.nsim and not args.force:
        return

    if args.force and condition.exists():
        shutil.rmtree(condition)

    condition.mkdir(parents=True, exist_ok=True)

    temporary_script = (
        condition / f"_temporary_{model}_beta{beta:.2f}.py"
    )

    make_bonus_model(source_file, temporary_script)

    command = [
        sys.executable,
        temporary_script.name,
        str(args.m),
        str(args.lr),
        str(args.tau),
        str(args.nsim),
        str(args.T),
        str(beta),
    ]

    log_path = condition / f"{model}_beta{beta:.2f}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=condition,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    temporary_script.unlink(missing_ok=True)

    if process.returncode != 0:
        raise RuntimeError(
            f"{model} beta={beta:.2f} failed. Check {log_path}."
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
            "CountOutcomeT must contain four columns."
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

def find_sims(directory):
    simulations = []

    for path in directory.rglob("CountOutcomeT-sim*.txt"):
        match = SIMULATION_PATTERN.fullmatch(path.name)

        if match:
            simulations.append(
                (int(match.group("simulation")), path.parent)
            )

    return sorted(simulations, key=lambda item: item[0])


# collect cooperation bonus results

def get_bonus_results(args):
    rows = []
    learning_curves = {}

    for model in args.models:
        for beta in args.beta_values:
            beta = float(beta)
            condition = get_condition_dir(args.output, model, beta)
            simulations = find_sims(condition)

            if not simulations:
                continue

            key = (model, beta)
            learning_curves.setdefault(key, [])

            for simulation, simulation_dir in simulations:
                outcomes_path = (
                    simulation_dir
                    / f"CountOutcomeT-sim{simulation:04d}.txt"
                )
                defections_path = (
                    simulation_dir
                    / f"AgentCountDT-sim{simulation:04d}.txt"
                )
                policy_path = (
                    simulation_dir
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
                n_rows = min(len(cooperation), defections.shape[0])

                cooperation = cooperation[:n_rows]
                defections = defections[:n_rows]

                final_length = max(
                    1,
                    math.ceil(n_rows * args.final_fraction),
                )
                final_slice = slice(n_rows - final_length, n_rows)

                final_cooperation = float(
                    np.nanmean(cooperation[final_slice])
                )

                defection_rates = (
                    np.mean(defections[final_slice], axis=0)
                    / float(args.rounds)
                )

                remaining_defectors = (
                    defection_rates >= args.defector_threshold
                )

                population = defections.shape[1]
                remaining_count = int(np.sum(remaining_defectors))
                remaining_fraction = remaining_count / population

                strategies = [np.nan] * 4

                if policy_path.exists():
                    policies = load_data(policy_path)
                    policy_rows = min(len(policies), n_rows)
                    policies = policies[:policy_rows]

                    policy_length = max(
                        1,
                        math.ceil(
                            policy_rows * args.final_fraction
                        ),
                    )

                    totals = policies.sum(axis=1, keepdims=True)
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
                        "cooperation_bonus": beta,
                        "simulation": simulation,
                        "population": population,
                        "final_cooperation": final_cooperation,
                        "final_noncooperation": 1.0 - final_cooperation,
                        "remaining_defector_count": remaining_count,
                        "remaining_defector_fraction": remaining_fraction,
                        "pd_always_cooperate": strategies[0],
                        "pd_conditional": strategies[1],
                        "pd_contrarian": strategies[2],
                        "pd_always_defect": strategies[3],
                    }
                )

                learning_curves[key].append(cooperation)

    if not rows:
        raise RuntimeError(
            "No complete cooperation-bonus simulations were found."
        )

    results = pd.DataFrame(rows).sort_values(
        ["model", "cooperation_bonus", "simulation"]
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

    standard_error = np.std(values, ddof=1) / np.sqrt(len(values))
    margin = 1.96 * standard_error

    return mean_value - margin, mean_value + margin


# calculate cohens d

def cohens_d(control, treatment):
    control = np.asarray(control, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)

    control = control[np.isfinite(control)]
    treatment = treatment[np.isfinite(treatment)]

    if len(control) < 2 or len(treatment) < 2:
        return np.nan

    pooled_variance = (
        (len(control) - 1) * np.var(control, ddof=1)
        + (len(treatment) - 1) * np.var(treatment, ddof=1)
    ) / (len(control) + len(treatment) - 2)

    if pooled_variance <= 0:
        return np.nan

    return (
        np.mean(treatment) - np.mean(control)
    ) / np.sqrt(pooled_variance)


# summarise each beta condition

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

    for (model, beta), group in results.groupby(
        ["model", "cooperation_bonus"]
    ):
        row = {
            "model": model,
            "cooperation_bonus": beta,
            "number_simulations": len(group),
        }

        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]

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
        .sort_values(["model", "cooperation_bonus"])
        .reset_index(drop=True)
    )


# compare each bonus against beta zero

def run_bonus_stats(results):
    metrics = [
        "final_cooperation",
        "final_noncooperation",
        "remaining_defector_fraction",
        "pd_always_cooperate",
        "pd_conditional",
        "pd_always_defect",
    ]

    rows = []

    for model in sorted(results["model"].unique()):
        model_results = results[
            results["model"] == model
        ]

        control = model_results[
            np.isclose(
                model_results["cooperation_bonus"],
                0.0,
            )
        ]

        if control.empty:
            continue

        betas = sorted(
            beta
            for beta in model_results[
                "cooperation_bonus"
            ].unique()
            if not np.isclose(beta, 0.0)
        )

        for beta in betas:
            treatment = model_results[
                np.isclose(
                    model_results["cooperation_bonus"],
                    beta,
                )
            ]

            for metric in metrics:
                control_values = (
                    control[metric]
                    .dropna()
                    .to_numpy(dtype=np.float64)
                )

                treatment_values = (
                    treatment[metric]
                    .dropna()
                    .to_numpy(dtype=np.float64)
                )

                if not len(control_values) or not len(treatment_values):
                    continue

                welch_t = welch_p = mann_u = mann_p = np.nan

                if (
                    SCIPY_AVAILABLE
                    and len(control_values) >= 2
                    and len(treatment_values) >= 2
                ):
                    welch = ttest_ind(
                        control_values,
                        treatment_values,
                        equal_var=False,
                        nan_policy="omit",
                    )

                    mann = mannwhitneyu(
                        control_values,
                        treatment_values,
                        alternative="two-sided",
                    )

                    welch_t = welch.statistic
                    welch_p = welch.pvalue
                    mann_u = mann.statistic
                    mann_p = mann.pvalue

                control_mean = np.mean(control_values)
                treatment_mean = np.mean(treatment_values)

                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "control_beta": 0.0,
                        "treatment_beta": beta,
                        "control_mean": control_mean,
                        "treatment_mean": treatment_mean,
                        "mean_change": treatment_mean - control_mean,
                        "welch_t_statistic": welch_t,
                        "welch_p_value": welch_p,
                        "mann_whitney_u": mann_u,
                        "mann_whitney_p_value": mann_p,
                        "cohens_d": cohens_d(
                            control_values,
                            treatment_values,
                        ),
                    }
                )

    return pd.DataFrame(rows)


# plot one final outcome against beta

def plot_bonus_results(
    summary,
    metric,
    ylabel,
    title,
    filename,
    output_directory,
    y_limits=None,
):
    figure, axis = plt.subplots(figsize=(9, 6))

    for model in ["baseline", "hybrid"]:
        model_data = summary[
            summary["model"] == model
        ].sort_values("cooperation_bonus")

        if model_data.empty:
            continue

        beta_values = model_data[
            "cooperation_bonus"
        ].to_numpy()

        means = model_data[
            f"mean_{metric}"
        ].to_numpy()

        lower = model_data[
            f"ci_lower_{metric}"
        ].to_numpy()

        upper = model_data[
            f"ci_upper_{metric}"
        ].to_numpy()

        axis.errorbar(
            beta_values,
            means,
            yerr=np.vstack([means - lower, upper - means]),
            color=MODEL_COLOURS[model],
            marker=MODEL_MARKERS[model],
            markersize=8,
            linewidth=2.6,
            capsize=5,
            label=model.capitalize(),
        )

    axis.set_title(title)
    axis.set_xlabel(r"Cooperation Bonus, $\beta$")
    axis.set_ylabel(ylabel)
    axis.set_xticks(
        sorted(summary["cooperation_bonus"].unique())
    )

    if y_limits:
        axis.set_ylim(*y_limits)

    axis.grid(alpha=0.25)
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
    for model in ["baseline", "hybrid"]:
        model_curves = {
            beta: curves
            for (curve_model, beta), curves
            in learning_curves.items()
            if curve_model == model
        }

        if not model_curves:
            continue

        figure, axis = plt.subplots(figsize=(10, 6))

        for index, beta in enumerate(sorted(model_curves)):
            curves = model_curves[beta]

            if not curves:
                continue

            minimum_length = min(
                len(curve)
                for curve in curves
            )

            matrix = np.vstack(
                [curve[:minimum_length] for curve in curves]
            )

            mean_curve = np.nanmean(matrix, axis=0)
            steps = np.arange(1, minimum_length + 1)

            if model == "baseline":
                colour = PURPLE if index % 2 == 0 else LIGHT_PURPLE
            else:
                colour = PINK if index % 2 == 0 else LIGHT_PINK

            axis.plot(
                steps,
                moving_average(mean_curve, window),
                color=colour,
                linestyle=BETA_LINE_STYLES.get(
                    round(float(beta), 2),
                    "-",
                ),
                linewidth=2.5,
                label=rf"$\beta={beta:.2f}$",
            )

        axis.set_title(
            f"Cooperation During Training: {model.capitalize()} Model"
        )
        axis.set_xlabel("Training Iteration")
        axis.set_ylabel("Cooperation")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(title="Cooperation bonus")

        figure.tight_layout()
        figure.savefig(
            output_directory
            / f"cooperation_learning_curves_{model}_by_bonus.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


# plot final strategy composition

def plot_strategy_composition(summary, output_directory):
    strategy_columns = [
        "mean_pd_always_cooperate",
        "mean_pd_conditional",
        "mean_pd_contrarian",
        "mean_pd_always_defect",
    ]

    for model in ["baseline", "hybrid"]:
        model_data = summary[
            summary["model"] == model
        ].sort_values("cooperation_bonus")

        if model_data.empty:
            continue

        beta_values = model_data[
            "cooperation_bonus"
        ].to_numpy()

        figure, axis = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(beta_values))

        for name, column, colour in zip(
            STRATEGY_NAMES,
            strategy_columns,
            STRATEGY_COLOURS,
        ):
            values = model_data[column].to_numpy()

            axis.bar(
                beta_values,
                values,
                bottom=bottom,
                width=0.07,
                color=colour,
                alpha=0.88,
                label=name,
            )

            bottom += np.nan_to_num(values)

        axis.set_title(
            f"Learned PD Strategies by Cooperation Bonus: "
            f"{model.capitalize()}"
        )
        axis.set_xlabel(r"Cooperation Bonus, $\beta$")
        axis.set_ylabel("Proportion of Agents")
        axis.set_xticks(beta_values)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

        figure.tight_layout()
        figure.savefig(
            output_directory
            / f"learned_strategies_{model}_by_bonus.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


# run cooperation bonus ablation

def main():
    args = get_args()

    if not args.beta_values:
        raise ValueError("At least one beta value is required.")

    if any(beta < 0 for beta in args.beta_values):
        raise ValueError("Cooperation bonuses must be non-negative.")

    if not any(np.isclose(beta, 0.0) for beta in args.beta_values):
        raise ValueError(
            "beta=0 must be included as the control."
        )

    if not 0 < args.final_fraction <= 1:
        raise ValueError(
            "--final-fraction must be between zero and one."
        )

    if not 0 <= args.defector_threshold <= 1:
        raise ValueError(
            "--defector-threshold must be between zero and one."
        )

    if args.nsim <= 0 or args.T <= 0:
        raise ValueError(
            "Number of simulations and training horizon must be positive."
        )

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    model_files = {
        "baseline": args.baseline_file,
        "hybrid": args.hybrid_file,
    }

    if not args.skip_runs:
        for model in args.models:
            for beta in args.beta_values:
                run_bonus_condition(
                    model_files[model],
                    model,
                    float(beta),
                    args,
                )

    analysis_dir = args.output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    results, learning_curves = get_bonus_results(args)
    summary = summarise_results(results)
    statistical_tests = run_bonus_stats(results)

    results.to_csv(
        analysis_dir
        / "cooperation_bonus_results_by_simulation.csv",
        index=False,
    )

    summary.to_csv(
        analysis_dir
        / "cooperation_bonus_summary.csv",
        index=False,
    )

    statistical_tests.to_csv(
        analysis_dir
        / "cooperation_bonus_statistical_tests.csv",
        index=False,
    )

    plot_bonus_results(
        summary,
        "final_cooperation",
        "Final Cooperation",
        "Effect of Cooperation Bonus on Final Cooperation",
        "final_cooperation_by_bonus.png",
        analysis_dir,
        (0, 1),
    )

    plot_bonus_results(
        summary,
        "remaining_defector_fraction",
        "Remaining Defector Proportion",
        "Remaining Defectors by Cooperation Bonus",
        "remaining_defectors_by_bonus.png",
        analysis_dir,
        (0, 1),
    )

    plot_bonus_results(
        summary,
        "final_noncooperation",
        "Shortfall from Complete Cooperation",
        "Remaining Shortfall from 100% Cooperation",
        "cooperation_shortfall_by_bonus.png",
        analysis_dir,
        (0, 1),
    )

    plot_learning_curves(
        learning_curves,
        analysis_dir,
        args.window,
    )

    plot_strategy_composition(
        summary,
        analysis_dir,
    )

    print("\nCooperation-bonus ablation completed.")
    print(f"Results saved in: {analysis_dir}")


if __name__ == "__main__":
    main()