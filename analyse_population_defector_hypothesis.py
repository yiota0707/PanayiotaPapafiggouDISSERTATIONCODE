# analyses how population size affects cooperation and residual defection
# measures persistent defectors, defector pairings, and whether cooperation plateaus between n=50 and n=100

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

try:
    from scipy.stats import mannwhitneyu, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# dissertation figure colours

PURPLE = "#7b6fd6"
PINK = "#c06082"

MODEL_COLOURS = {
    "baseline": PURPLE,
    "hybrid": PINK,
}

MODEL_MARKERS = {
    "baseline": "o",
    "hybrid": "s",
}


# file patterns

MODEL_PATTERN = re.compile(
    r"(?:^|[_-])(?P<model>baseline|hybrid)(?:[_-]|$)"
)
POPULATION_PATTERN = re.compile(
    r"(?:^|_)Nagent(?P<population>\d+)(?:_|$)"
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
        description="Analyse residual defection across population sizes."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing population experiment outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("population_defector_analysis"),
        help="Directory used for analysis outputs.",
    )
    parser.add_argument(
        "--population-sizes",
        nargs="+",
        type=int,
        default=[10, 20, 50, 100],
        help="Population sizes to analyse.",
    )
    parser.add_argument(
        "--m-values",
        nargs="+",
        type=float,
        default=[1.0],
        help="Assortativity values to analyse.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "hybrid"],
        default=["baseline", "hybrid"],
        help="Models to analyse.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Repeated-game rounds per training iteration.",
    )
    parser.add_argument(
        "--final-fraction",
        type=float,
        default=0.10,
        help="Final fraction of training used for measurements.",
    )
    parser.add_argument(
        "--defector-threshold",
        type=float,
        default=0.50,
        help="Defection-rate threshold used to classify defectors.",
    )
    parser.add_argument(
        "--persistence-threshold",
        type=float,
        default=0.90,
        help="Fraction of the final window required for persistent defection.",
    )
    parser.add_argument(
        "--plateau-difference",
        type=float,
        default=0.02,
        help="Maximum practical difference between N=50 and N=100.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level used for intervals.",
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


# build a simulation output path

def get_output_path(
    simulation_directory: Path,
    prefix: str,
    simulation: int,
    extension: str,
) -> Path:
    return simulation_directory / f"{prefix}-sim{simulation:04d}.{extension}"


# calculate cooperation

def get_cooperation(outcomes: np.ndarray) -> np.ndarray:
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
        out=np.full_like(cooperative_actions, np.nan, dtype=np.float64),
        where=total_actions > 0,
    )


# confidence interval helpers

def get_critical_value(confidence_level: float) -> float:
    if np.isclose(confidence_level, 0.90):
        return 1.645
    if np.isclose(confidence_level, 0.95):
        return 1.960
    if np.isclose(confidence_level, 0.99):
        return 2.576

    return 1.960


def get_ci(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    mean_value = float(np.mean(values))

    if len(values) == 1:
        return mean_value, mean_value

    standard_error = float(
        np.std(values, ddof=1) / np.sqrt(len(values))
    )
    critical_value = get_critical_value(confidence_level)

    return (
        mean_value - critical_value * standard_error,
        mean_value + critical_value * standard_error,
    )


# calculate cohens d

def cohens_d(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]

    if len(first) < 2 or len(second) < 2:
        return np.nan

    first_variance = np.var(first, ddof=1)
    second_variance = np.var(second, ddof=1)

    pooled_variance = (
        (len(first) - 1) * first_variance
        + (len(second) - 1) * second_variance
    ) / (len(first) + len(second) - 2)

    if pooled_variance <= 0:
        return np.nan

    return float(
        (np.mean(first) - np.mean(second))
        / math.sqrt(pooled_variance)
    )


# find population result folders

def find_result_directories(
    root: Path,
    models: list[str],
    population_sizes: list[int],
    m_values: list[float],
) -> list[Path]:
    excluded_terms = (
        "__pycache__",
        "archive",
        "backup",
        "old_results",
        "component_ablation",
        "switch_learning_ablation",
        "learning_rate",
        "lr_ablation",
        "tau_ablation",
        "horizon_ablation",
    )

    result_directories = []

    for directory in root.rglob("result_*"):
        if not directory.is_dir():
            continue

        folder_path = str(directory).lower()

        if any(term in folder_path for term in excluded_terms):
            continue

        name = directory.name
        model_match = MODEL_PATTERN.search(name)
        population_match = POPULATION_PATTERN.search(name)
        m_match = M_PATTERN.search(name)

        if (
            model_match is None
            or population_match is None
            or m_match is None
        ):
            continue

        model = model_match.group("model")
        population = int(population_match.group("population"))
        m_value = float(m_match.group("m"))

        model_selected = model in models
        population_selected = population in population_sizes
        m_selected = any(
            np.isclose(m_value, requested)
            for requested in m_values
        )
        main_parameters = "_lr0.05_" in name and "_R20_" in name

        if (
            model_selected
            and population_selected
            and m_selected
            and main_parameters
        ):
            result_directories.append(directory)

    result_directories = sorted(set(result_directories))

    if not result_directories:
        raise FileNotFoundError(
            "No population-size result directories were found."
        )

    return result_directories


# read experiment settings from folder name

def get_folder_info(
    directory: Path,
) -> tuple[str, int, float]:
    model_match = MODEL_PATTERN.search(directory.name)
    population_match = POPULATION_PATTERN.search(directory.name)
    m_match = M_PATTERN.search(directory.name)

    if (
        model_match is None
        or population_match is None
        or m_match is None
    ):
        raise ValueError(
            f"Could not read experiment settings from {directory}"
        )

    model = model_match.group("model")
    population = int(population_match.group("population"))
    m_value = float(m_match.group("m"))

    return model, population, m_value


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


# calculate remaining and persistent defectors

def get_defector_measures(
    agent_defections: np.ndarray,
    rounds: int,
    final_fraction: float,
    defector_threshold: float,
    persistence_threshold: float,
) -> dict[str, float | np.ndarray]:
    n_steps, population = agent_defections.shape

    final_window = max(
        1,
        int(math.ceil(n_steps * final_fraction)),
    )

    final_period = agent_defections[-final_window:]
    defection_rates = final_period / float(rounds)

    mean_defection_rate = np.mean(defection_rates, axis=0)
    remaining_defectors = mean_defection_rate >= defector_threshold

    defective_each_iteration = defection_rates >= defector_threshold
    defective_time_fraction = np.mean(
        defective_each_iteration,
        axis=0,
    )
    persistent_defectors = (
        defective_time_fraction >= persistence_threshold
    )

    remaining_count = int(np.sum(remaining_defectors))
    persistent_count = int(np.sum(persistent_defectors))

    return {
        "remaining_defector_count": remaining_count,
        "remaining_defector_fraction": remaining_count / float(population),
        "persistent_defector_count": persistent_count,
        "persistent_defector_fraction": persistent_count / float(population),
        "mean_agent_defection_rate": float(np.mean(mean_defection_rate)),
        "maximum_agent_defection_rate": float(np.max(mean_defection_rate)),
        "remaining_defector_mask": remaining_defectors,
        "persistent_defector_mask": persistent_defectors,
    }


# analyse final defector pairings

def get_pair_measures(
    final_groups: np.ndarray,
    final_actions: np.ndarray,
    persistent_defector_mask: np.ndarray,
) -> dict[str, float]:
    final_groups = np.asarray(final_groups, dtype=np.int64)
    final_actions = np.asarray(final_actions, dtype=np.int64).reshape(-1)

    if final_groups.ndim != 2 or final_groups.shape[1] != 2:
        return {
            "final_dd_pair_count": np.nan,
            "final_dd_pair_fraction": np.nan,
            "agents_in_final_dd_pairs": np.nan,
            "agent_fraction_in_final_dd_pairs": np.nan,
            "persistent_dd_pair_count": np.nan,
            "persistent_defectors_paired_together_fraction": np.nan,
        }

    n_pairs = final_groups.shape[0]
    population = len(final_actions)

    dd_pair_count = 0
    persistent_dd_pair_count = 0
    persistent_agents_in_dd_pairs = set()

    for first_agent, second_agent in final_groups:
        first_agent = int(first_agent)
        second_agent = int(second_agent)

        if (
            first_agent < 0
            or second_agent < 0
            or first_agent >= population
            or second_agent >= population
        ):
            continue

        first_defected = final_actions[first_agent] == 1
        second_defected = final_actions[second_agent] == 1

        if first_defected and second_defected:
            dd_pair_count += 1

        first_persistent = bool(
            persistent_defector_mask[first_agent]
        )
        second_persistent = bool(
            persistent_defector_mask[second_agent]
        )

        if first_persistent and second_persistent:
            persistent_dd_pair_count += 1
            persistent_agents_in_dd_pairs.add(first_agent)
            persistent_agents_in_dd_pairs.add(second_agent)

    persistent_count = int(np.sum(persistent_defector_mask))
    agents_in_dd_pairs = 2 * dd_pair_count

    return {
        "final_dd_pair_count": dd_pair_count,
        "final_dd_pair_fraction": (
            dd_pair_count / float(n_pairs)
            if n_pairs > 0
            else np.nan
        ),
        "agents_in_final_dd_pairs": agents_in_dd_pairs,
        "agent_fraction_in_final_dd_pairs": (
            agents_in_dd_pairs / float(population)
            if population > 0
            else np.nan
        ),
        "persistent_dd_pair_count": persistent_dd_pair_count,
        "persistent_defectors_paired_together_fraction": (
            len(persistent_agents_in_dd_pairs) / float(persistent_count)
            if persistent_count > 0
            else 0.0
        ),
    }


# compare two groups

def compare_groups(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]

    output = {
        "mean_first": float(np.mean(first)) if len(first) else np.nan,
        "mean_second": float(np.mean(second)) if len(second) else np.nan,
        "mean_difference": np.nan,
        "absolute_mean_difference": np.nan,
        "welch_t_statistic": np.nan,
        "welch_p_value": np.nan,
        "mann_whitney_u": np.nan,
        "mann_whitney_p_value": np.nan,
        "cohens_d": np.nan,
    }

    if len(first) == 0 or len(second) == 0:
        return output

    mean_difference = float(
        np.mean(second) - np.mean(first)
    )

    output["mean_difference"] = mean_difference
    output["absolute_mean_difference"] = abs(mean_difference)
    output["cohens_d"] = cohens_d(second, first)

    if SCIPY_AVAILABLE and len(first) >= 2 and len(second) >= 2:
        welch_test = ttest_ind(
            first,
            second,
            equal_var=False,
            nan_policy="omit",
        )

        mann_whitney = mannwhitneyu(
            first,
            second,
            alternative="two-sided",
        )

        output["welch_t_statistic"] = float(welch_test.statistic)
        output["welch_p_value"] = float(welch_test.pvalue)
        output["mann_whitney_u"] = float(mann_whitney.statistic)
        output["mann_whitney_p_value"] = float(mann_whitney.pvalue)

    return output


# compare consecutive population sizes

def run_pairwise_tests(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "final_cooperation",
        "remaining_defector_fraction",
        "persistent_defector_fraction",
        "persistent_defector_count",
        "final_dd_pair_fraction",
        "persistent_defectors_paired_together_fraction",
    ]

    rows = []

    for (model, m_value), condition in results.groupby(["model", "m"]):
        populations = sorted(condition["population"].unique())

        for first_population, second_population in zip(
            populations[:-1],
            populations[1:],
        ):
            first_group = condition[
                condition["population"] == first_population
            ]
            second_group = condition[
                condition["population"] == second_population
            ]

            for metric in metrics:
                comparison = compare_groups(
                    first_group[metric].to_numpy(),
                    second_group[metric].to_numpy(),
                )

                rows.append(
                    {
                        "model": model,
                        "m": m_value,
                        "first_population": first_population,
                        "second_population": second_population,
                        "metric": metric,
                        **comparison,
                    }
                )

    return pd.DataFrame(rows)


# compare n=50 and n=100

def run_plateau_tests(
    results: pd.DataFrame,
    plateau_threshold: float,
) -> pd.DataFrame:
    rows = []

    for (model, m_value), condition in results.groupby(["model", "m"]):
        n50_values = condition.loc[
            condition["population"] == 50,
            "final_cooperation",
        ].to_numpy()

        n100_values = condition.loc[
            condition["population"] == 100,
            "final_cooperation",
        ].to_numpy()

        if len(n50_values) == 0 or len(n100_values) == 0:
            continue

        comparison = compare_groups(
            n50_values,
            n100_values,
        )

        practical_plateau = (
            comparison["absolute_mean_difference"]
            <= plateau_threshold
        )

        statistical_plateau = (
            np.isfinite(comparison["welch_p_value"])
            and comparison["welch_p_value"] >= 0.05
        )

        rows.append(
            {
                "model": model,
                "m": m_value,
                "population_50_mean": comparison["mean_first"],
                "population_100_mean": comparison["mean_second"],
                "cooperation_gain_50_to_100": comparison["mean_difference"],
                "absolute_difference": comparison["absolute_mean_difference"],
                "welch_t_statistic": comparison["welch_t_statistic"],
                "welch_p_value": comparison["welch_p_value"],
                "mann_whitney_u": comparison["mann_whitney_u"],
                "mann_whitney_p_value": comparison["mann_whitney_p_value"],
                "cohens_d": comparison["cohens_d"],
                "practical_plateau": practical_plateau,
                "statistical_plateau": statistical_plateau,
                "combined_plateau_evidence": (
                    practical_plateau
                    and statistical_plateau
                ),
            }
        )

    return pd.DataFrame(rows)


# summarise population conditions

def summarise_conditions(
    results: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    metrics = [
        "final_cooperation",
        "remaining_defector_fraction",
        "remaining_defector_count",
        "persistent_defector_fraction",
        "persistent_defector_count",
        "final_dd_pair_fraction",
        "agent_fraction_in_final_dd_pairs",
        "persistent_dd_pair_count",
        "persistent_defectors_paired_together_fraction",
    ]

    rows = []

    for (
        model,
        m_value,
        population,
    ), group in results.groupby(
        ["model", "m", "population"]
    ):
        row = {
            "model": model,
            "m": m_value,
            "population": population,
            "number_simulations": len(group),
        }

        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            valid = values[np.isfinite(values)]

            if len(valid) == 0:
                row[f"mean_{metric}"] = np.nan
                row[f"sd_{metric}"] = np.nan
                row[f"ci_lower_{metric}"] = np.nan
                row[f"ci_upper_{metric}"] = np.nan
                continue

            lower, upper = get_ci(
                valid,
                confidence_level,
            )

            row[f"mean_{metric}"] = float(np.mean(valid))
            row[f"sd_{metric}"] = (
                float(np.std(valid, ddof=1))
                if len(valid) > 1
                else 0.0
            )
            row[f"ci_lower_{metric}"] = lower
            row[f"ci_upper_{metric}"] = upper

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["model", "m", "population"])
        .reset_index(drop=True)
    )


# plot a result across population sizes

def plot_population_results(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    output_directory: Path,
    y_limits: tuple[float, float] | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))

    for model in ["baseline", "hybrid"]:
        model_data = summary[
            summary["model"] == model
        ].sort_values("population")

        if model_data.empty:
            continue

        populations = model_data["population"].to_numpy()
        means = model_data[f"mean_{metric}"].to_numpy()
        lower = model_data[f"ci_lower_{metric}"].to_numpy()
        upper = model_data[f"ci_upper_{metric}"].to_numpy()

        axis.errorbar(
            populations,
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
    axis.set_xlabel("Population Size, $N$")
    axis.set_ylabel(ylabel)
    axis.set_xticks(sorted(summary["population"].unique()))

    if y_limits is not None:
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


# compare n=50 and n=100 cooperation

def plot_cooperation_plateau(
    results: pd.DataFrame,
    output_directory: Path,
) -> None:
    conditions = []
    values = []
    colours = []

    for model in ["baseline", "hybrid"]:
        for population in [50, 100]:
            selected = results.loc[
                (results["model"] == model)
                & (results["population"] == population),
                "final_cooperation",
            ].dropna()

            if selected.empty:
                continue

            conditions.append(
                f"{model.capitalize()}\n$N={population}$"
            )
            values.append(selected.to_numpy())
            colours.append(MODEL_COLOURS[model])

    if not values:
        return

    figure, axis = plt.subplots(figsize=(9, 6))

    boxplot = axis.boxplot(
        values,
        labels=conditions,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )

    for patch, colour in zip(boxplot["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)

    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.7)

    axis.set_title(
        "Cooperation Plateau Comparison: $N=50$ and $N=100$"
    )
    axis.set_ylabel("Final Cooperation")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "cooperation_plateau_N50_N100.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# run the population analysis

def analyse_results(
    args: argparse.Namespace,
) -> None:
    if not 0.0 < args.final_fraction <= 1.0:
        raise ValueError(
            "--final-fraction must be between 0 and 1."
        )

    if not 0.0 <= args.defector_threshold <= 1.0:
        raise ValueError(
            "--defector-threshold must be between 0 and 1."
        )

    if not 0.0 <= args.persistence_threshold <= 1.0:
        raise ValueError(
            "--persistence-threshold must be between 0 and 1."
        )

    if args.rounds <= 0:
        raise ValueError(
            "--rounds must be greater than zero."
        )

    root = args.root.resolve()
    output_directory = args.output.resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_directories = find_result_directories(
        root,
        args.models,
        args.population_sizes,
        args.m_values,
    )

    rows = []

    for result_directory in result_directories:
        model, population, m_value = get_folder_info(
            result_directory
        )

        for simulation, simulation_directory in find_simulations(
            result_directory
        ):
            outcomes_path = get_output_path(
                simulation_directory,
                "CountOutcomeT",
                simulation,
                "txt",
            )

            agent_defections_path = get_output_path(
                simulation_directory,
                "AgentCountDT",
                simulation,
                "txt",
            )

            final_groups_path = get_output_path(
                simulation_directory,
                "FinalGroups",
                simulation,
                "npy",
            )

            final_actions_path = get_output_path(
                simulation_directory,
                "FinalActionsPD",
                simulation,
                "npy",
            )

            if (
                not outcomes_path.exists()
                or not agent_defections_path.exists()
            ):
                continue

            try:
                outcomes = load_data(outcomes_path)
                agent_defections = load_data(
                    agent_defections_path
                )
                cooperation = get_cooperation(outcomes)

            except (
                OSError,
                ValueError,
            ) as error:
                print(
                    f"Skipping simulation {simulation:04d}: {error}"
                )
                continue

            current_population = agent_defections.shape[1]

            if current_population != population:
                print(
                    f"Warning: expected N={population}, "
                    f"found {current_population} agents."
                )

            final_window = max(
                1,
                int(
                    math.ceil(
                        len(cooperation)
                        * args.final_fraction
                    )
                ),
            )

            final_cooperation = float(
                np.nanmean(
                    cooperation[-final_window:]
                )
            )

            defector_measures = get_defector_measures(
                agent_defections,
                args.rounds,
                args.final_fraction,
                args.defector_threshold,
                args.persistence_threshold,
            )

            pair_measures = {
                "final_dd_pair_count": np.nan,
                "final_dd_pair_fraction": np.nan,
                "agents_in_final_dd_pairs": np.nan,
                "agent_fraction_in_final_dd_pairs": np.nan,
                "persistent_dd_pair_count": np.nan,
                "persistent_defectors_paired_together_fraction": np.nan,
            }

            if (
                final_groups_path.exists()
                and final_actions_path.exists()
            ):
                try:
                    pair_measures = get_pair_measures(
                        np.load(final_groups_path),
                        np.load(final_actions_path),
                        defector_measures[
                            "persistent_defector_mask"
                        ],
                    )

                except (
                    OSError,
                    ValueError,
                    IndexError,
                ) as error:
                    print(
                        f"Could not analyse final pairs "
                        f"for simulation {simulation:04d}: {error}"
                    )

            persistent_count = defector_measures[
                "persistent_defector_count"
            ]

            rows.append(
                {
                    "model": model,
                    "m": m_value,
                    "population": current_population,
                    "simulation": simulation,
                    "final_cooperation": final_cooperation,
                    "final_noncooperation": 1.0 - final_cooperation,
                    "remaining_defector_count": defector_measures[
                        "remaining_defector_count"
                    ],
                    "remaining_defector_fraction": defector_measures[
                        "remaining_defector_fraction"
                    ],
                    "persistent_defector_count": persistent_count,
                    "persistent_defector_fraction": defector_measures[
                        "persistent_defector_fraction"
                    ],
                    "mean_agent_defection_rate": defector_measures[
                        "mean_agent_defection_rate"
                    ],
                    "maximum_agent_defection_rate": defector_measures[
                        "maximum_agent_defection_rate"
                    ],
                    "small_persistent_defector_group": (
                        0 < persistent_count <= 4
                    ),
                    **pair_measures,
                }
            )

            print(
                f"{model:8s} "
                f"N={current_population:3d} "
                f"m={m_value:.2f} "
                f"sim={simulation:04d} "
                f"cooperation={final_cooperation:.4f} "
                f"persistent={persistent_count}"
            )

    if not rows:
        raise RuntimeError(
            "No complete population-size simulations were analysed."
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "model",
                "m",
                "population",
                "simulation",
            ]
        )
    )

    results.to_csv(
        output_directory
        / "population_results_by_simulation.csv",
        index=False,
    )

    summary = summarise_conditions(
        results,
        args.confidence_level,
    )

    summary.to_csv(
        output_directory
        / "population_summary_by_condition.csv",
        index=False,
    )

    pairwise_tests = run_pairwise_tests(results)

    pairwise_tests.to_csv(
        output_directory
        / "population_pairwise_tests.csv",
        index=False,
    )

    plateau_tests = run_plateau_tests(
        results,
        args.plateau_difference,
    )

    plateau_tests.to_csv(
        output_directory
        / "population_plateau_tests.csv",
        index=False,
    )

    # create figures

    plot_population_results(
        summary,
        "final_cooperation",
        "Final Cooperation",
        "Final Cooperation by Population Size",
        "cooperation_by_population.png",
        output_directory,
        (0.0, 1.0),
    )

    plot_population_results(
        summary,
        "remaining_defector_fraction",
        "Remaining Defector Proportion",
        "Remaining Defectors by Population Size",
        "remaining_defectors_by_population.png",
        output_directory,
        (0.0, 1.0),
    )

    plot_population_results(
        summary,
        "persistent_defector_count",
        "Number of Persistent Defectors",
        "Persistent Defector Count by Population Size",
        "persistent_defector_count_by_population.png",
        output_directory,
    )

    plot_population_results(
        summary,
        "final_dd_pair_fraction",
        "Proportion of Final Pairs that are D--D",
        "Final Defector--Defector Pairing by Population Size",
        "final_dd_pairing_by_population.png",
        output_directory,
        (0.0, 1.0),
    )

    plot_cooperation_plateau(
        results,
        output_directory,
    )

    print("\nPopulation analysis completed.")
    print(f"Results saved in: {output_directory}")

    if not SCIPY_AVAILABLE:
        print(
            "SciPy was not available, so statistical p-values "
            "were not calculated."
        )


if __name__ == "__main__":
    args = get_args()
    analyse_results(args)