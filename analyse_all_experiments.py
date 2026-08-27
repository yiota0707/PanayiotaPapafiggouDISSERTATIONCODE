# analyses the main assortativity and parameter sensitivity experiments
# summarises final cooperation across conditions and generates comparison figures

import glob
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# plot colours

PINK = "#c06082"
PURPLE = "#7b6fd6"


# read experiment settings from a result folder

def get_folder_info(folder):
    model = "hybrid" if "_hybrid_" in folder else "baseline"

    lr = np.nan
    n_agents = np.nan
    m = np.nan
    tau = np.nan
    horizon = np.nan

    match = re.search(r"_lr([0-9]+\.[0-9]+)", folder)
    if match:
        lr = float(match.group(1))

    match = re.search(r"_Nagent([0-9]+)", folder)
    if match:
        n_agents = int(match.group(1))

    match = re.search(r"_m([0-9]+\.[0-9]+)", folder)
    if match:
        m = float(match.group(1))

    match = re.search(r"tau([0-9]+\.[0-9]+)", folder)
    if match:
        tau = float(match.group(1))

    match = re.search(r"T([0-9]+)", folder)
    if match:
        horizon = int(match.group(1))

    # main assortativity experiment is the default
    experiment = "main_m_ablation"
    parameter = "m"
    parameter_value = m

    if "ablation_tau" in folder:
        experiment = "temperature_ablation"
        parameter = "tau"
        parameter_value = tau
    elif "ablation_horizon" in folder:
        experiment = "horizon_ablation"
        parameter = "T"
        parameter_value = horizon
    elif not np.isnan(lr) and lr != 0.05:
        experiment = "learning_rate_ablation"
        parameter = "lr"
        parameter_value = lr
    elif not np.isnan(n_agents) and n_agents != 20:
        experiment = "population_ablation"
        parameter = "Nagent"
        parameter_value = n_agents

    return {
        "model": model,
        "experiment": experiment,
        "parameter": parameter,
        "parameter_value": parameter_value,
        "lr": lr,
        "Nagent": n_agents,
        "m": m,
        "tau": tau,
        "T": horizon,
    }


# calculate final cooperation

def get_final_cooperation(path):
    outcomes = np.loadtxt(path, delimiter=",")

    if outcomes.ndim == 1:
        outcomes = outcomes.reshape(1, -1)

    cc, cd, dc, dd = outcomes[-1, :4]
    total_pairs = cc + cd + dc + dd

    if total_pairs <= 0:
        return np.nan

    # cc gives two cooperative actions and cd and dc give one each
    return (2 * cc + cd + dc) / (2 * total_pairs)


# collect results from all main sensitivity experiments

def get_experiment_data():
    folders = []

    folders.extend(
        glob.glob("./result_PD2_Qpast1-b-*")
    )

    folders.extend(
        glob.glob(
            "./ablation_tau/tau*/**/result_PD2_Qpast1-b-*",
            recursive=True,
        )
    )

    folders.extend(
        glob.glob(
            "./ablation_horizon/T*/**/result_PD2_Qpast1-b-*",
            recursive=True,
        )
    )

    rows = []

    for folder in sorted(set(folders)):
        if "baseline" not in folder and "hybrid" not in folder:
            continue

        folder_info = get_folder_info(folder)

        files = sorted(
            glob.glob(
                os.path.join(
                    folder,
                    "sim*",
                    "CountOutcomeT-sim*.txt",
                )
            )
        )

        for file_path in files:
            sim_match = re.search(r"sim(\d+)", file_path)
            simulation = int(sim_match.group(1)) if sim_match else np.nan

            final_cooperation = get_final_cooperation(file_path)

            if np.isnan(final_cooperation):
                continue

            rows.append(
                {
                    "folder": folder,
                    **folder_info,
                    "sim": simulation,
                    "final_cooperation": final_cooperation,
                    "source_file": file_path,
                }
            )

    return pd.DataFrame(rows)


# summarise each experimental condition

def summarise_experiments(data):
    group_columns = [
        "experiment",
        "parameter",
        "parameter_value",
        "model",
        "lr",
        "Nagent",
        "m",
        "tau",
        "T",
    ]

    return (
        data.groupby(
            group_columns,
            dropna=False,
        )["final_cooperation"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values(
            ["experiment", "parameter_value", "model"]
        )
    )


# plot one experiment

def plot_experiment_results(
    summary,
    experiment,
    filename,
    title,
    xlabel,
):
    experiment_data = summary[
        summary["experiment"] == experiment
    ].copy()

    if experiment_data.empty:
        return

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(7, 5))

    for model, colour in [
        ("baseline", PINK),
        ("hybrid", PURPLE),
    ]:
        model_data = experiment_data[
            experiment_data["model"] == model
        ].sort_values("parameter_value")

        if model_data.empty:
            continue

        plt.errorbar(
            model_data["parameter_value"],
            model_data["mean"],
            yerr=model_data["std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=model.capitalize(),
            color=colour,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Final cooperation rate")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join("figures", filename),
        dpi=300,
    )

    plt.close()


# run experiment analysis

def main():
    data = get_experiment_data()

    if data.empty:
        print("No experiment data found.")
        return

    data.to_csv(
        "all_experiments_raw_values.csv",
        index=False,
    )

    summary = summarise_experiments(data)

    summary.to_csv(
        "all_experiments_summary.csv",
        index=False,
    )

    experiments = [
        (
            "main_m_ablation",
            "main_m_ablation.png",
            "Main m-ablation",
            "Assortativity m",
        ),
        (
            "learning_rate_ablation",
            "learning_rate_ablation.png",
            "Learning-rate ablation",
            "Learning rate",
        ),
        (
            "population_ablation",
            "population_ablation.png",
            "Population ablation",
            "Number of agents",
        ),
        (
            "horizon_ablation",
            "horizon_ablation.png",
            "Horizon ablation",
            "Training horizon T",
        ),
        (
            "temperature_ablation",
            "temperature_ablation.png",
            "Temperature ablation",
            "Temperature tau",
        ),
    ]

    for experiment, filename, title, xlabel in experiments:
        plot_experiment_results(
            summary,
            experiment,
            filename,
            title,
            xlabel,
        )

    print("\nExperiment analysis completed.")
    print("Saved: all_experiments_raw_values.csv")
    print("Saved: all_experiments_summary.csv")


if __name__ == "__main__":
    main()