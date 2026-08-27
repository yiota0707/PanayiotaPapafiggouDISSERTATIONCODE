# runs the training-horizon sensitivity experiment for baseline and hybrid models
# compares cooperation across different training lengths at m=0 and m=1

import os
import subprocess

SCRIPTS_DIR = os.getcwd()

# experiment settings

T_VALUES = [50000, 100000, 200000]
M_VALUES = [0.0, 1.0]

LR = 0.05
TAU = 1.0
NSIM = 10

BASELINE_SCRIPT = os.path.join(
    SCRIPTS_DIR,
    "MAgame2106s2_baseline_assort_m_njit.py",
)

HYBRID_SCRIPT = os.path.join(
    SCRIPTS_DIR,
    "MAgame2106s2_hybrid_assort_m_njit.py",
)


# run one training horizon condition

def run_condition(script, model, training_horizon, m):
    output_dir = os.path.join(
        SCRIPTS_DIR,
        "ablation_horizon",
        f"T{training_horizon}",
        f"{model}_m{m:.2f}",
    )
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = SCRIPTS_DIR + ":" + env.get("PYTHONPATH", "")

    subprocess.run(
        [
            "python",
            "-u",
            script,
            str(m),
            str(LR),
            str(TAU),
            str(NSIM),
            str(training_horizon),
        ],
        cwd=output_dir,
        env=env,
        check=True,
    )

    print(f"Completed: {model}, T={training_horizon}, m={m}")


# run training horizon sensitivity analysis

def main():
    models = [
        ("baseline", BASELINE_SCRIPT),
        ("hybrid", HYBRID_SCRIPT),
    ]

    for training_horizon in T_VALUES:
        for m in M_VALUES:
            for model, script in models:
                run_condition(
                    script,
                    model,
                    training_horizon,
                    m,
                )

    print("\nTraining horizon sensitivity analysis completed.")


if __name__ == "__main__":
    main()