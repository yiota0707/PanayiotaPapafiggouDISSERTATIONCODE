# analyses behavioural dynamics in the baseline and hybrid models
# examines cooperation, switching, stability, strategy evolution and behavioural correlations

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dissertation figure colours

PURPLE = "#7b6fd6"
PINK = "#c06082"
LIGHT_PURPLE = "#b9b2ee"
LIGHT_PINK = "#e6a8bd"
DARK_PURPLE = "#5547b8"
DARK_PINK = "#913f61"

STRATEGY_COLOURS = [PINK, PURPLE, LIGHT_PINK, LIGHT_PURPLE]

STRATEGY_NAMES = [
    "Always Cooperate",
    "Conditional",
    "Contrarian",
    "Always Defect",
]

SWITCH_STRATEGY_NAMES = [
    "Always Stay",
    "Conditional",
    "Contrarian",
    "Always Switch",
]


# result folder patterns

MODEL_PATTERN = re.compile(
    r"(?:^|[_-])(?P<model>baseline|hybrid)(?:[_-]|$)"
)
M_PATTERN = re.compile(
    r"(?:^|_)m(?P<m>\d+(?:\.\d+)?)"
)
SIMULATION_PATTERN = re.compile(
    r"CountOutcomeT-sim(?P<simulation>\d+)\.txt$"
)


# command line settings

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse behavioural dynamics of baseline and hybrid models."
    )

    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="Root directory containing result folders.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("behavioural_dynamics_analysis"),
        help="Directory used for analysis outputs.",
    )
    parser.add_argument(
        "--m-values", nargs="+", type=float, default=[0.0, 1.0],
        help="Assortativity values to analyse.",
    )
    parser.add_argument(
        "--rounds", type=int, default=20,
        help="Number of repeated-game rounds.",
    )
    parser.add_argument(
        "--window", type=int, default=2000,
        help="Rolling smoothing window used in figures.",
    )
    parser.add_argument(
        "--final-fraction", type=float, default=0.10,
        help="Final fraction of training used for summaries.",
    )
    parser.add_argument(
        "--behaviour-threshold", type=float, default=0.50,
        help="Threshold used to classify an agent as defective.",
    )
    parser.add_argument(
        "--stability-window", type=int, default=5000,
        help="Window used to detect behavioural stability.",
    )
    parser.add_argument(
        "--stability-threshold", type=float, default=0.01,
        help="Maximum behavioural change allowed for stability.",
    )

    return parser.parse_args()


# load simulation data

def load_data(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, delimiter=",")
    except ValueError:
        values = np.loadtxt(path)

    values = np.asarray(values, dtype=np.float64)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    return values


# smooth a time series

def smooth_data(values: np.ndarray, window: int) -> np.ndarray:
    window_size = max(1, min(int(window), len(values)))

    return (
        pd.Series(values)
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


# calculate cooperation from game outcomes

def get_cooperation(outcomes: np.ndarray) -> np.ndarray:
    if outcomes.shape[1] != 4:
        raise ValueError(
            "CountOutcomeT must contain four columns: CC, CD, DC and DD."
        )

    cc, cd, dc, dd = outcomes.T

    cooperative_actions = 2.0 * cc + cd + dc
    total_actions = 2.0 * (cc + cd + dc + dd)

    if np.any(total_actions == 0):
        raise ValueError(
            "CountOutcomeT contains an iteration with zero outcomes."
        )

    return cooperative_actions / total_actions


# convert state labels to q table indices

def get_state_index(state: int) -> int:
    if state == 0:
        return 0
    if state == 1:
        return 1
    if state == 100:
        return 2
    if state == 101:
        return 3

    raise ValueError(f"Unknown state label: {state}")


# classify final learned strategy

def get_final_strategy(
    q_values: np.ndarray,
    state_c: int,
    state_d: int,
) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)

    if q_values.ndim != 3:
        raise ValueError(
            "FinalQ must have shape "
            "(number_agents, number_states, number_actions)."
        )

    if q_values.shape[1] < 4 or q_values.shape[2] < 2:
        raise ValueError("FinalQ does not contain the required states and actions.")

    state_c_index = get_state_index(state_c)
    state_d_index = get_state_index(state_d)

    action_zero_after_c = (
        q_values[:, state_c_index, 0]
        > q_values[:, state_c_index, 1]
    )
    action_zero_after_d = (
        q_values[:, state_d_index, 0]
        > q_values[:, state_d_index, 1]
    )

    strategies = np.full(q_values.shape[0], 3, dtype=np.int64)

    strategies[action_zero_after_c & action_zero_after_d] = 0
    strategies[action_zero_after_c & ~action_zero_after_d] = 1
    strategies[~action_zero_after_c & action_zero_after_d] = 2
    strategies[~action_zero_after_c & ~action_zero_after_d] = 3

    return strategies


# find main baseline and hybrid result folders

def find_result_directories(
    root: Path,
    m_values: list[float],
) -> list[Path]:
    excluded_terms = (
        "ablation",
        "component",
        "switch_learning",
        "population",
        "horizon",
        "temperature",
        "tau",
        "archive",
        "backup",
        "__pycache__",
    )

    result_directories = []

    for directory in root.rglob("result_*"):
        if not directory.is_dir():
            continue

        name = directory.name
        folder_path = str(directory).lower()

        if any(term in folder_path for term in excluded_terms):
            continue

        main_settings = (
            "_lr0.05_" in name
            and "_Nagent20_" in name
            and "_R20_" in name
        )

        if not main_settings:
            continue

        model_match = MODEL_PATTERN.search(name)
        m_match = M_PATTERN.search(name)

        if model_match is None or m_match is None:
            continue

        model = model_match.group("model")
        m_value = float(m_match.group("m"))

        requested_m = any(
            np.isclose(m_value, requested)
            for requested in m_values
        )

        if model in ("baseline", "hybrid") and requested_m:
            result_directories.append(directory)

    result_directories = sorted(set(result_directories))

    if not result_directories:
        raise FileNotFoundError(
            "No matching main result directories were found."
        )

    return result_directories


# read model and assortativity from folder name

def get_folder_info(directory: Path) -> tuple[str, float]:
    model_match = MODEL_PATTERN.search(directory.name)
    m_match = M_PATTERN.search(directory.name)

    if model_match is None or m_match is None:
        raise ValueError(
            f"Could not read experiment settings from {directory}"
        )

    return model_match.group("model"), float(m_match.group("m"))


# find completed simulations

def find_simulations(
    result_directory: Path,
) -> list[tuple[int, Path]]:
    simulations = []

    for path in result_directory.rglob("CountOutcomeT-sim*.txt"):
        match = SIMULATION_PATTERN.fullmatch(path.name)

        if match is None:
            continue

        simulation = int(match.group("simulation"))
        simulations.append((simulation, path.parent))

    return sorted(simulations, key=lambda item: item[0])


# build a simulation output path

def get_output_path(
    simulation_directory: Path,
    prefix: str,
    simulation: int,
    extension: str = "txt",
) -> Path:
    return (
        simulation_directory
        / f"{prefix}-sim{simulation:04d}.{extension}"
    )


# convert strategy counts to proportions

def get_strategy_proportions(
    strategy_counts: np.ndarray,
) -> np.ndarray:
    totals = strategy_counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0

    return strategy_counts / totals


# calculate behavioural transitions

def get_behaviour_transitions(
    agent_defections: np.ndarray,
    rounds: int,
    threshold: float,
) -> dict[str, np.ndarray]:
    defection_rates = agent_defections / float(rounds)
    defective_state = defection_rates >= threshold

    n_steps = defective_state.shape[0]

    cooperative_to_defective = np.zeros(n_steps, dtype=np.float64)
    defective_to_cooperative = np.zeros(n_steps, dtype=np.float64)
    unchanged_fraction = np.zeros(n_steps, dtype=np.float64)

    unchanged_fraction[0] = 1.0

    if n_steps > 1:
        previous = defective_state[:-1]
        current = defective_state[1:]

        cooperative_to_defective[1:] = np.mean(
            (~previous) & current,
            axis=1,
        )
        defective_to_cooperative[1:] = np.mean(
            previous & (~current),
            axis=1,
        )
        unchanged_fraction[1:] = np.mean(
            previous == current,
            axis=1,
        )

    behaviour_change_frequency = (
        cooperative_to_defective
        + defective_to_cooperative
    )

    return {
        "defection_rates": defection_rates,
        "defective_state": defective_state,
        "defective_fraction": np.mean(defective_state, axis=1),
        "cooperative_to_defective": cooperative_to_defective,
        "defective_to_cooperative": defective_to_cooperative,
        "behaviour_change_frequency": behaviour_change_frequency,
        "unchanged_fraction": unchanged_fraction,
    }


# calculate behavioural run lengths

def get_run_lengths(
    defective_state: np.ndarray,
) -> tuple[float, float, float]:
    cooperative_runs = []
    defective_runs = []
    all_runs = []

    n_steps, n_agents = defective_state.shape

    if n_steps == 0:
        return np.nan, np.nan, np.nan

    for agent in range(n_agents):
        states = defective_state[:, agent].astype(np.int8)

        current_state = states[0]
        run_length = 1

        for state in states[1:]:
            if state == current_state:
                run_length += 1
                continue

            all_runs.append(run_length)

            if current_state == 0:
                cooperative_runs.append(run_length)
            else:
                defective_runs.append(run_length)

            current_state = state
            run_length = 1

        all_runs.append(run_length)

        if current_state == 0:
            cooperative_runs.append(run_length)
        else:
            defective_runs.append(run_length)

    mean_cooperative = (
        float(np.mean(cooperative_runs))
        if cooperative_runs
        else np.nan
    )
    mean_defective = (
        float(np.mean(defective_runs))
        if defective_runs
        else np.nan
    )
    mean_overall = (
        float(np.mean(all_runs))
        if all_runs
        else np.nan
    )

    return mean_cooperative, mean_defective, mean_overall


# find when behaviour becomes stable

def get_stability_time(
    change_frequency: np.ndarray,
    stability_window: int,
    stability_threshold: float,
) -> float:
    if len(change_frequency) == 0:
        return np.nan

    window_size = min(
        max(1, stability_window),
        len(change_frequency),
    )

    rolling_change = (
        pd.Series(change_frequency)
        .rolling(
            window=window_size,
            min_periods=window_size,
        )
        .mean()
        .to_numpy(dtype=np.float64)
    )

    stable_indices = np.where(
        rolling_change <= stability_threshold
    )[0]

    if len(stable_indices) == 0:
        return np.nan

    first_completed_window = int(stable_indices[0])
    first_window_start = first_completed_window - window_size + 1

    return float(first_window_start + 1)


# calculate correlation safely

def get_correlation(
    x: pd.Series,
    y: pd.Series,
    method: str,
) -> float:
    valid = x.notna() & y.notna()

    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) < 3:
        return np.nan

    if x_valid.nunique() < 2 or y_valid.nunique() < 2:
        return np.nan

    return float(
        x_valid.corr(
            y_valid,
            method=method,
        )
    )


# prepare mean strategy curve

def get_mean_curve(curves: list[np.ndarray]) -> np.ndarray:
    minimum_length = min(curve.shape[0] for curve in curves)

    matrix = np.stack(
        [curve[:minimum_length] for curve in curves],
        axis=0,
    )

    return np.mean(matrix, axis=0)


# plot learned pd strategies

def plot_strategy_evolution(
    strategy_curves: dict,
    output_directory: Path,
    window: int,
) -> None:
    line_styles = ["-", "--", "-.", ":"]

    for (model, m_value), curves in strategy_curves.items():
        if not curves:
            continue

        mean_curve = get_mean_curve(curves)
        n_steps = mean_curve.shape[0]
        steps = np.arange(1, n_steps + 1)

        figure, axis = plt.subplots(figsize=(10, 6))

        for strategy_index in range(4):
            axis.plot(
                steps,
                smooth_data(
                    mean_curve[:, strategy_index],
                    window,
                ),
                color=STRATEGY_COLOURS[strategy_index],
                linewidth=2.4,
                linestyle=line_styles[strategy_index],
                label=STRATEGY_NAMES[strategy_index],
            )

        axis.set_title(
            f"Learned PD Strategies: {model.capitalize()}, "
            f"$m={m_value:.2f}$"
        )
        axis.set_xlabel("Training Iteration")
        axis.set_ylabel("Proportion of Agents")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)

        figure.tight_layout()

        figure.savefig(
            output_directory
            / f"strategy_evolution_{model}_m{m_value:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# plot learned switching strategies

def plot_switch_strategy_evolution(
    strategy_curves: dict,
    output_directory: Path,
    window: int,
) -> None:
    line_styles = ["-", "--", "-.", ":"]

    for (model, m_value), curves in strategy_curves.items():
        if not curves:
            continue

        mean_curve = get_mean_curve(curves)
        n_steps = mean_curve.shape[0]
        steps = np.arange(1, n_steps + 1)

        figure, axis = plt.subplots(figsize=(10, 6))

        for strategy_index in range(4):
            axis.plot(
                steps,
                smooth_data(
                    mean_curve[:, strategy_index],
                    window,
                ),
                color=STRATEGY_COLOURS[strategy_index],
                linewidth=2.4,
                linestyle=line_styles[strategy_index],
                label=SWITCH_STRATEGY_NAMES[strategy_index],
            )

        axis.set_title(
            f"Learned Switching Strategies: {model.capitalize()}, "
            f"$m={m_value:.2f}$"
        )
        axis.set_xlabel("Training Iteration")
        axis.set_ylabel("Proportion of Agents")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)

        figure.tight_layout()

        figure.savefig(
            output_directory
            / f"switch_strategy_evolution_{model}_m{m_value:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# plot behavioural transitions

def plot_behavioural_transitions(
    transition_curves: dict,
    output_directory: Path,
    window: int,
) -> None:
    for (model, m_value), curves in transition_curves.items():
        if not curves:
            continue

        n_steps = min(
            len(curve["cooperative_to_defective"])
            for curve in curves
        )

        cooperative_to_defective = np.mean(
            np.vstack(
                [
                    curve["cooperative_to_defective"][:n_steps]
                    for curve in curves
                ]
            ),
            axis=0,
        )

        defective_to_cooperative = np.mean(
            np.vstack(
                [
                    curve["defective_to_cooperative"][:n_steps]
                    for curve in curves
                ]
            ),
            axis=0,
        )

        total_changes = np.mean(
            np.vstack(
                [
                    curve["behaviour_change_frequency"][:n_steps]
                    for curve in curves
                ]
            ),
            axis=0,
        )

        steps = np.arange(1, n_steps + 1)

        figure, axis = plt.subplots(figsize=(10, 6))

        axis.plot(
            steps,
            smooth_data(cooperative_to_defective, window),
            color=PURPLE,
            linewidth=2.4,
            label="Cooperative to Defective",
        )

        axis.plot(
            steps,
            smooth_data(defective_to_cooperative, window),
            color=PINK,
            linewidth=2.4,
            label="Defective to Cooperative",
        )

        axis.plot(
            steps,
            smooth_data(total_changes, window),
            color=DARK_PINK,
            linewidth=2.4,
            linestyle="--",
            label="Total Behavioural Change",
        )

        axis.set_title(
            f"Behavioural Transitions: {model.capitalize()}, "
            f"$m={m_value:.2f}$"
        )
        axis.set_xlabel("Training Iteration")
        axis.set_ylabel("Proportion of Agents Changing State")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
        axis.legend()

        figure.tight_layout()

        figure.savefig(
            output_directory
            / f"behavioural_transitions_{model}_m{m_value:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# plot switching and cooperation

def plot_switching_and_cooperation(
    time_series: dict,
    output_directory: Path,
    window: int,
) -> None:
    for (model, m_value), curves in time_series.items():
        if not curves:
            continue

        n_steps = min(
            len(curve["cooperation"])
            for curve in curves
        )

        cooperation = np.mean(
            np.vstack(
                [
                    curve["cooperation"][:n_steps]
                    for curve in curves
                ]
            ),
            axis=0,
        )

        switching = np.mean(
            np.vstack(
                [
                    curve["switching_rate"][:n_steps]
                    for curve in curves
                ]
            ),
            axis=0,
        )

        steps = np.arange(1, n_steps + 1)

        figure, axis = plt.subplots(figsize=(10, 6))

        axis.plot(
            steps,
            smooth_data(cooperation, window),
            color=PINK,
            linewidth=2.7,
            label="Cooperation",
        )

        axis.plot(
            steps,
            smooth_data(switching, window),
            color=PURPLE,
            linewidth=2.5,
            linestyle="--",
            label="Switching Frequency",
        )

        axis.set_title(
            f"Switching and Cooperation: {model.capitalize()}, "
            f"$m={m_value:.2f}$"
        )
        axis.set_xlabel("Training Iteration")
        axis.set_ylabel("Rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()

        if model == "baseline":
            axis.text(
                0.02,
                0.04,
                "Baseline rematching is imposed by the model design.",
                transform=axis.transAxes,
                fontsize=9,
                color=DARK_PURPLE,
            )

        figure.tight_layout()

        figure.savefig(
            output_directory
            / f"switching_and_cooperation_{model}_m{m_value:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# plot behavioural stability

def plot_behavioural_stability(
    time_series: dict,
    output_directory: Path,
    window: int,
) -> None:
    for (model, m_value), curves in time_series.items():
        if not curves:
            continue

        n_steps = min(
            len(curve["unchanged_fraction"])
            for curve in curves
        )

        unchanged = np.mean(
            np.vstack(
                [
                    curve["unchanged_fraction"][:n_steps]
                    for curve in curves
                ]
            ),
            axis=0,
        )

        change = np.mean(
            np.vstack(
                [
                    curve["behaviour_change_frequency"][:n_steps]
                    for curve in curves
                ]
            ),
            axis=0,
        )

        steps = np.arange(1, n_steps + 1)

        figure, axis = plt.subplots(figsize=(10, 6))

        axis.plot(
            steps,
            smooth_data(unchanged, window),
            color=PINK,
            linewidth=2.6,
            label="Behavioural Stability",
        )

        axis.plot(
            steps,
            smooth_data(change, window),
            color=PURPLE,
            linewidth=2.4,
            linestyle="--",
            label="Behavioural Change",
        )

        axis.set_title(
            f"Behavioural Stability: {model.capitalize()}, "
            f"$m={m_value:.2f}$"
        )
        axis.set_xlabel("Training Iteration")
        axis.set_ylabel("Proportion of Agents")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()

        figure.tight_layout()

        figure.savefig(
            output_directory
            / f"behavioural_stability_{model}_m{m_value:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# plot behavioural correlations

def plot_correlation_ranking(
    correlations: pd.DataFrame,
    output_directory: Path,
) -> None:
    for (model, m_value), group in correlations.groupby(["model", "m"]):
        valid = group.dropna(
            subset=["spearman_correlation"]
        ).copy()

        if valid.empty:
            continue

        valid["absolute_correlation"] = (
            valid["spearman_correlation"].abs()
        )

        valid = valid.sort_values(
            "absolute_correlation",
            ascending=True,
        )

        bar_colours = [
            PINK if value >= 0 else PURPLE
            for value in valid["spearman_correlation"]
        ]

        figure, axis = plt.subplots(figsize=(10, 8))

        axis.barh(
            valid["factor"],
            valid["spearman_correlation"],
            color=bar_colours,
            alpha=0.88,
        )

        axis.axvline(
            0.0,
            color=DARK_PURPLE,
            linewidth=1.2,
        )

        axis.set_title(
            f"Behavioural Factors Related to Cooperation: "
            f"{model.capitalize()}, $m={m_value:.2f}$"
        )

        axis.set_xlabel(
            "Spearman Correlation with Final Cooperation"
        )

        axis.grid(axis="x", alpha=0.25)

        figure.tight_layout()

        figure.savefig(
            output_directory
            / f"correlation_ranking_{model}_m{m_value:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# run behavioural analysis

def analyse_results(args: argparse.Namespace) -> None:
    if not 0.0 < args.final_fraction <= 1.0:
        raise ValueError(
            "--final-fraction must be between 0 and 1."
        )

    if not 0.0 <= args.behaviour_threshold <= 1.0:
        raise ValueError(
            "--behaviour-threshold must be between 0 and 1."
        )

    if args.rounds <= 0:
        raise ValueError("--rounds must be greater than zero.")

    if args.stability_window <= 0:
        raise ValueError(
            "--stability-window must be greater than zero."
        )

    if args.stability_threshold < 0:
        raise ValueError(
            "--stability-threshold cannot be negative."
        )

    root = args.root.resolve()
    output_directory = args.output.resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_directories = find_result_directories(
        root,
        args.m_values,
    )

    simulation_data = []
    agent_data = []

    pd_strategy_curves = {}
    switch_strategy_curves = {}
    transition_curves = {}
    time_series = {}

    for result_directory in result_directories:
        model, m_value = get_folder_info(result_directory)
        key = (model, m_value)

        pd_strategy_curves.setdefault(key, [])
        switch_strategy_curves.setdefault(key, [])
        transition_curves.setdefault(key, [])
        time_series.setdefault(key, [])

        for simulation, simulation_directory in find_simulations(
            result_directory
        ):
            files = {
                "outcomes": get_output_path(
                    simulation_directory,
                    "CountOutcomeT",
                    simulation,
                ),
                "switches": get_output_path(
                    simulation_directory,
                    "CountSwitchT",
                    simulation,
                ),
                "pd_policy": get_output_path(
                    simulation_directory,
                    "CountPolicyPDTypeT",
                    simulation,
                ),
                "switch_policy": get_output_path(
                    simulation_directory,
                    "CountPolicySwTypeT",
                    simulation,
                ),
                "agent_switches": get_output_path(
                    simulation_directory,
                    "AgentCountSwT",
                    simulation,
                ),
                "agent_defections": get_output_path(
                    simulation_directory,
                    "AgentCountDT",
                    simulation,
                ),
                "final_q": get_output_path(
                    simulation_directory,
                    "FinalQ",
                    simulation,
                    "npy",
                ),
            }

            missing = [
                name
                for name, path in files.items()
                if not path.exists()
            ]

            if missing:
                continue

            try:
                outcomes = load_data(files["outcomes"])
                switch_counts = load_data(files["switches"])
                pd_policy_counts = load_data(files["pd_policy"])
                switch_policy_counts = load_data(files["switch_policy"])
                agent_switch_counts = load_data(files["agent_switches"])
                agent_defection_counts = load_data(files["agent_defections"])
                final_q = np.load(files["final_q"])

            except (
                OSError,
                ValueError,
            ) as error:
                print(
                    f"Skipping simulation {simulation}: {error}"
                )
                continue

            cooperation = get_cooperation(outcomes)

            n_steps = min(
                len(cooperation),
                switch_counts.shape[0],
                pd_policy_counts.shape[0],
                switch_policy_counts.shape[0],
                agent_switch_counts.shape[0],
                agent_defection_counts.shape[0],
            )

            cooperation = cooperation[:n_steps]
            switch_counts = switch_counts[:n_steps]
            pd_policy_counts = pd_policy_counts[:n_steps]
            switch_policy_counts = switch_policy_counts[:n_steps]
            agent_switch_counts = agent_switch_counts[:n_steps]
            agent_defection_counts = agent_defection_counts[:n_steps]

            switch_total = switch_counts.sum(axis=1)

            switch_rate = np.divide(
                switch_counts[:, 1],
                switch_total,
                out=np.zeros_like(
                    switch_total,
                    dtype=np.float64,
                ),
                where=switch_total > 0,
            )

            pd_strategy_proportions = get_strategy_proportions(
                pd_policy_counts
            )
            switch_strategy_proportions = get_strategy_proportions(
                switch_policy_counts
            )

            transitions = get_behaviour_transitions(
                agent_defection_counts,
                args.rounds,
                args.behaviour_threshold,
            )

            (
                mean_cooperative_run_length,
                mean_defective_run_length,
                mean_overall_run_length,
            ) = get_run_lengths(
                transitions["defective_state"]
            )

            time_to_stability = get_stability_time(
                transitions["behaviour_change_frequency"],
                args.stability_window,
                args.stability_threshold,
            )

            final_window = max(
                1,
                int(
                    math.ceil(
                        n_steps
                        * args.final_fraction
                    )
                ),
            )

            final_period = slice(
                n_steps - final_window,
                n_steps,
            )

            final_cooperation = float(
                np.mean(cooperation[final_period])
            )

            final_switching = float(
                np.mean(switch_rate[final_period])
            )

            final_defective_fraction = float(
                np.mean(
                    transitions["defective_fraction"][final_period]
                )
            )

            cooperative_to_defective_frequency = float(
                np.mean(
                    transitions["cooperative_to_defective"][final_period]
                )
            )

            defective_to_cooperative_frequency = float(
                np.mean(
                    transitions["defective_to_cooperative"][final_period]
                )
            )

            behaviour_change_frequency = float(
                np.mean(
                    transitions["behaviour_change_frequency"][final_period]
                )
            )

            behavioural_stability = float(
                np.mean(
                    transitions["unchanged_fraction"][final_period]
                )
            )

            final_pd_strategy = np.mean(
                pd_strategy_proportions[final_period],
                axis=0,
            )

            final_switch_strategy = np.mean(
                switch_strategy_proportions[final_period],
                axis=0,
            )

            simulation_data.append(
                {
                    "model": model,
                    "m": m_value,
                    "simulation": simulation,
                    "final_cooperation": final_cooperation,
                    "final_switching_frequency": final_switching,
                    "final_defective_fraction": final_defective_fraction,
                    "cooperative_to_defective_frequency":
                        cooperative_to_defective_frequency,
                    "defective_to_cooperative_frequency":
                        defective_to_cooperative_frequency,
                    "behaviour_change_frequency":
                        behaviour_change_frequency,
                    "behavioural_stability":
                        behavioural_stability,
                    "mean_cooperative_run_length":
                        mean_cooperative_run_length,
                    "mean_defective_run_length":
                        mean_defective_run_length,
                    "mean_overall_run_length":
                        mean_overall_run_length,
                    "time_to_behavioural_stability":
                        time_to_stability,
                    "pd_always_cooperate": final_pd_strategy[0],
                    "pd_conditional": final_pd_strategy[1],
                    "pd_contrarian": final_pd_strategy[2],
                    "pd_always_defect": final_pd_strategy[3],
                    "switch_always_stay": final_switch_strategy[0],
                    "switch_conditional": final_switch_strategy[1],
                    "switch_contrarian": final_switch_strategy[2],
                    "switch_always_switch": final_switch_strategy[3],
                }
            )

            pd_strategy_curves[key].append(
                pd_strategy_proportions
            )

            switch_strategy_curves[key].append(
                switch_strategy_proportions
            )

            transition_curves[key].append(
                transitions
            )

            time_series[key].append(
                {
                    "cooperation": cooperation,
                    "switching_rate": switch_rate,
                    "unchanged_fraction":
                        transitions["unchanged_fraction"],
                    "behaviour_change_frequency":
                        transitions["behaviour_change_frequency"],
                }
            )

            agent_defection_rates = (
                np.mean(
                    agent_defection_counts[final_period],
                    axis=0,
                )
                / float(args.rounds)
            )

            agent_cooperation_rates = (
                1.0 - agent_defection_rates
            )

            agent_switching_rates = (
                np.mean(
                    agent_switch_counts[final_period],
                    axis=0,
                )
                / float(args.rounds)
            )

            final_pd_types = get_final_strategy(
                final_q,
                100,
                101,
            )

            final_switch_types = get_final_strategy(
                final_q,
                0,
                1,
            )

            n_agents = min(
                final_q.shape[0],
                len(agent_cooperation_rates),
                len(agent_switching_rates),
            )

            for agent in range(n_agents):
                agent_data.append(
                    {
                        "model": model,
                        "m": m_value,
                        "simulation": simulation,
                        "agent": agent,
                        "cooperation_rate":
                            agent_cooperation_rates[agent],
                        "defection_rate":
                            agent_defection_rates[agent],
                        "switching_rate":
                            agent_switching_rates[agent],
                        "final_pd_strategy":
                            STRATEGY_NAMES[final_pd_types[agent]],
                        "final_switch_strategy":
                            SWITCH_STRATEGY_NAMES[
                                final_switch_types[agent]
                            ],
                    }
                )

            print(
                f"{model:8s} "
                f"m={m_value:.2f} "
                f"sim={simulation:04d} "
                f"cooperation={final_cooperation:.4f} "
                f"switching={final_switching:.4f}"
            )

    if not simulation_data:
        raise RuntimeError(
            "No complete simulations were analysed."
        )

    simulation_results = pd.DataFrame(
        simulation_data
    )
    agent_results = pd.DataFrame(
        agent_data
    )

    simulation_results.to_csv(
        output_directory
        / "behavioural_summary_by_simulation.csv",
        index=False,
    )

    agent_results.to_csv(
        output_directory
        / "agent_behaviour.csv",
        index=False,
    )

    # summarise each condition

    summary = (
        simulation_results
        .groupby(["model", "m"])
        .agg(
            number_simulations=("simulation", "count"),
            mean_final_cooperation=("final_cooperation", "mean"),
            sd_final_cooperation=("final_cooperation", "std"),
            mean_switching_frequency=("final_switching_frequency", "mean"),
            mean_defective_fraction=("final_defective_fraction", "mean"),
            mean_cooperative_to_defective=(
                "cooperative_to_defective_frequency",
                "mean",
            ),
            mean_defective_to_cooperative=(
                "defective_to_cooperative_frequency",
                "mean",
            ),
            mean_behaviour_change_frequency=(
                "behaviour_change_frequency",
                "mean",
            ),
            mean_behavioural_stability=("behavioural_stability", "mean"),
            mean_cooperative_run_length=(
                "mean_cooperative_run_length",
                "mean",
            ),
            mean_defective_run_length=(
                "mean_defective_run_length",
                "mean",
            ),
            mean_overall_run_length=("mean_overall_run_length", "mean"),
            mean_time_to_behavioural_stability=(
                "time_to_behavioural_stability",
                "mean",
            ),
            mean_pd_always_cooperate=("pd_always_cooperate", "mean"),
            mean_pd_conditional=("pd_conditional", "mean"),
            mean_pd_contrarian=("pd_contrarian", "mean"),
            mean_pd_always_defect=("pd_always_defect", "mean"),
            mean_switch_always_stay=("switch_always_stay", "mean"),
            mean_switch_conditional=("switch_conditional", "mean"),
            mean_switch_contrarian=("switch_contrarian", "mean"),
            mean_switch_always_switch=("switch_always_switch", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        output_directory
        / "behavioural_summary_by_condition.csv",
        index=False,
    )

    # calculate behavioural correlations

    behavioural_factors = [
        "final_switching_frequency",
        "final_defective_fraction",
        "cooperative_to_defective_frequency",
        "defective_to_cooperative_frequency",
        "behaviour_change_frequency",
        "behavioural_stability",
        "mean_cooperative_run_length",
        "mean_defective_run_length",
        "mean_overall_run_length",
        "time_to_behavioural_stability",
        "pd_always_cooperate",
        "pd_conditional",
        "pd_contrarian",
        "pd_always_defect",
        "switch_always_stay",
        "switch_conditional",
        "switch_contrarian",
        "switch_always_switch",
    ]

    correlation_data = []

    for (model, m_value), group in simulation_results.groupby(
        ["model", "m"]
    ):
        for factor in behavioural_factors:
            pearson = get_correlation(
                group[factor],
                group["final_cooperation"],
                "pearson",
            )

            spearman = get_correlation(
                group[factor],
                group["final_cooperation"],
                "spearman",
            )

            correlation_data.append(
                {
                    "model": model,
                    "m": m_value,
                    "factor": factor,
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                    "absolute_spearman": (
                        abs(spearman)
                        if np.isfinite(spearman)
                        else np.nan
                    ),
                }
            )

    correlations = (
        pd.DataFrame(correlation_data)
        .sort_values(
            ["model", "m", "absolute_spearman"],
            ascending=[True, True, False],
        )
    )

    correlations.to_csv(
        output_directory
        / "behavioural_correlations.csv",
        index=False,
    )

    # create figures

    plot_strategy_evolution(
        pd_strategy_curves,
        output_directory,
        args.window,
    )

    plot_switch_strategy_evolution(
        switch_strategy_curves,
        output_directory,
        args.window,
    )

    plot_behavioural_transitions(
        transition_curves,
        output_directory,
        args.window,
    )

    plot_switching_and_cooperation(
        time_series,
        output_directory,
        args.window,
    )

    plot_behavioural_stability(
        time_series,
        output_directory,
        args.window,
    )

    plot_correlation_ranking(
        correlations,
        output_directory,
    )

    print("\nBehavioural dynamics analysis completed.")
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    args = get_args()
    analyse_results(args)