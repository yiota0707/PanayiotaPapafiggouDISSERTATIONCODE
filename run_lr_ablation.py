# runs the learning-rate sensitivity experiment for baseline and hybrid models
# compares cooperation across different learning rates at m=0 and m=1

from MAgame2106s2_baseline_assort_m_njit import run_experiment_batch as run_baseline
from MAgame2106s2_hybrid_assort_m_njit import run_experiment_batch as run_hybrid


# experiment settings

LR_VALUES = [0.01, 0.05, 0.10, 0.20]
M_VALUES = [0.0, 1.0]

GAME_NAME = "PD2"
NACT = 2
ROUNDS = 20
TAU = 1.0
NAGENT = 20
NSIM = 10
T = 100000
SEED = 1234


# run learning rate sensitivity analysis

def main():
    models = [
        ("baseline", run_baseline, "Qpast1-b-baseline"),
        ("hybrid", run_hybrid, "Qpast1-b-hybrid"),
    ]

    for learning_rate in LR_VALUES:
        for m in M_VALUES:
            for model, run_model, algorithm in models:
                run_model(
                    gameName=GAME_NAME,
                    Nact=NACT,
                    roundsG=ROUNDS,
                    algoName=algorithm,
                    lr=learning_rate,
                    Nagent=NAGENT,
                    simStart=1,
                    Nsim=NSIM,
                    tStart=0,
                    T=T,
                    m=m,
                    seed=SEED,
                    tau=TAU,
                )

                print(f"Completed: {model}, lr={learning_rate}, m={m}")

    print("\nLearning rate sensitivity analysis completed.")


if __name__ == "__main__":
    main()