# performs statistical comparisons between baseline and hybrid across experimental conditions
# calculates confidence intervals, welch tests, mann-whitney tests and cohen's d effect sizes

import os
import numpy as np
import pandas as pd
from scipy import stats


# input and output files

INPUT = "all_experiments_raw_values.csv"

OUTPUT_DIR = "statistical_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "validated_cooperation_values.csv",
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "validated_statistical_summary.csv",
)

LATEX_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "validated_statistical_table.tex",
)


# calculate confidence interval

def get_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan, np.nan

    mean_value = np.mean(values)
    standard_error = stats.sem(values)

    margin = (
        standard_error
        * stats.t.ppf(
            (1 + confidence) / 2,
            len(values) - 1,
        )
    )

    return mean_value - margin, mean_value + margin


# calculate cohens d

def cohens_d(baseline, hybrid):
    baseline = np.asarray(baseline, dtype=float)
    hybrid = np.asarray(hybrid, dtype=float)

    n_baseline = len(baseline)
    n_hybrid = len(hybrid)

    if n_baseline < 2 or n_hybrid < 2:
        return np.nan

    pooled_sd = np.sqrt(
        (
            (n_baseline - 1) * np.var(baseline, ddof=1)
            + (n_hybrid - 1) * np.var(hybrid, ddof=1)
        )
        / (n_baseline + n_hybrid - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (
        np.mean(hybrid) - np.mean(baseline)
    ) / pooled_sd


# compare baseline and hybrid for each condition

def run_statistical_tests(data):
    rows = []

    group_columns = [
        "experiment",
        "parameter",
        "parameter_value",
    ]

    for group_key, group in data.groupby(
        group_columns,
        dropna=False,
    ):
        experiment, parameter, parameter_value = group_key

        baseline = group.loc[
            group["model"] == "baseline",
            "final_cooperation",
        ].dropna().to_numpy()

        hybrid = group.loc[
            group["model"] == "hybrid",
            "final_cooperation",
        ].dropna().to_numpy()

        if len(baseline) == 0 or len(hybrid) == 0:
            continue

        baseline_ci_low, baseline_ci_high = get_ci(baseline)
        hybrid_ci_low, hybrid_ci_high = get_ci(hybrid)

        welch_test = stats.ttest_ind(
            hybrid,
            baseline,
            equal_var=False,
        )

        mann_whitney = stats.mannwhitneyu(
            hybrid,
            baseline,
            alternative="two-sided",
        )

        effect_size = cohens_d(
            baseline,
            hybrid,
        )

        rows.append(
            {
                "experiment": experiment,
                "parameter": parameter,
                "parameter_value": parameter_value,
                "baseline_n": len(baseline),
                "hybrid_n": len(hybrid),
                "baseline_mean": np.mean(baseline),
                "baseline_std": np.std(baseline, ddof=1),
                "baseline_ci95_low": baseline_ci_low,
                "baseline_ci95_high": baseline_ci_high,
                "hybrid_mean": np.mean(hybrid),
                "hybrid_std": np.std(hybrid, ddof=1),
                "hybrid_ci95_low": hybrid_ci_low,
                "hybrid_ci95_high": hybrid_ci_high,
                "difference_hybrid_minus_baseline":
                    np.mean(hybrid) - np.mean(baseline),
                "welch_t_stat": welch_test.statistic,
                "welch_p_value": welch_test.pvalue,
                "mannwhitney_u": mann_whitney.statistic,
                "mannwhitney_p_value": mann_whitney.pvalue,
                "cohens_d": effect_size,
                "significant_p_lt_0.05":
                    bool(welch_test.pvalue < 0.05),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "experiment",
            "parameter_value",
        ]
    )


# save compact latex table

def save_latex_table(stats_summary):
    columns = [
        "experiment",
        "parameter",
        "parameter_value",
        "baseline_mean",
        "hybrid_mean",
        "difference_hybrid_minus_baseline",
        "welch_p_value",
        "cohens_d",
        "significant_p_lt_0.05",
    ]

    latex_table = stats_summary[
        columns
    ].copy()

    latex_table["baseline_mean"] = (
        latex_table["baseline_mean"]
        .round(3)
    )

    latex_table["hybrid_mean"] = (
        latex_table["hybrid_mean"]
        .round(3)
    )

    latex_table["difference_hybrid_minus_baseline"] = (
        latex_table["difference_hybrid_minus_baseline"]
        .round(3)
    )

    latex_table["welch_p_value"] = (
        latex_table["welch_p_value"]
        .map(lambda value: f"{value:.3g}")
    )

    latex_table["cohens_d"] = (
        latex_table["cohens_d"]
        .round(3)
    )

    latex_table.to_latex(
        LATEX_OUTPUT,
        index=False,
        escape=False,
    )


# run statistical analysis

def main():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(
            f"{INPUT} not found. "
            "Run analyse_all_experiments.py first."
        )

    data = pd.read_csv(INPUT)

    required_columns = {
        "experiment",
        "parameter",
        "parameter_value",
        "model",
        "final_cooperation",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns from {INPUT}: "
            f"{missing_columns}"
        )

    data.to_csv(
        RAW_OUTPUT,
        index=False,
    )

    stats_summary = run_statistical_tests(
        data
    )

    stats_summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    save_latex_table(
        stats_summary
    )

    print("\nStatistical analysis completed.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()