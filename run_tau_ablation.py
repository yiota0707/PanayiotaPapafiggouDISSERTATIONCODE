# runs the temperature sensitivity experiment for baseline and hybrid models
# compares cooperation across tau=0.1, 0.5 and 1.0 at m=0 and m=1

import os
import subprocess

SCRIPTS_DIR = os.getcwd()

# experiment settings

TAU_VALUES = [0.1, 0.5, 1.0]
M_VALUES = [0.0, 1.0]

LR = 0.05
NSIM = 10
T = 100000

BASELINE_SCRIPT = os.path.join(
    SCRIPTS_DIR,
    "MAgame2106s2_baseline_assort_m_njit.py",
)

HYBRID_SCRIPT = os.path.join(
    SCRIPTS_DIR,
    "MAgame2106s2_hybrid_assort_m_njit.py",
)


# run one temperature condition

def run_condition(script, model, tau, m):
    output_dir = os.path.join(
        SCRIPTS_DIR,
        "ablation_tau",
        f"tau{tau:.2f}",
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
            str(tau),
            str(NSIM),
            str(T),
        ],
        cwd=output_dir,
        env=env,
        check=True,
    )

    print(f"Completed: {model}, tau={tau}, m={m}")


# run temperature sensitivity analysis

def main():
    models = [
        ("baseline", BASELINE_SCRIPT),
        ("hybrid", HYBRID_SCRIPT),
    ]

    for tau in TAU_VALUES:
        for m in M_VALUES:
            for model, script in models:
                run_condition(
                    script,
                    model,
                    tau,
                    m,
                )

    print("\nTemperature sensitivity analysis completed.")


if __name__ == "__main__":
    main()