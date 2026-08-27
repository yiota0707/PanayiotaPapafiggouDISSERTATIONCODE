# runs the main assortativity experiment for baseline and hybrid models
# compares cooperation across m=0, 0.3, 0.5, 0.7 and 1 under the default settings

from MAgame2106s2_baseline_assort_m_njit import run_experiment_batch as run_baseline
from MAgame2106s2_hybrid_assort_m_njit import run_experiment_batch as run_hybrid


# experiment settings

M_VALUES = [0.00, 0.30, 0.50, 0.70, 1.00]

GAME_NAME = "PD2"
NACT = 2
ROUNDS = 20
LR = 0.05
TAU = 1.0
NAGENT = 20
NSIM = 10
T = 100000
SEED = 1234


# run main assortativity experiment

def main():
    models = [
        ("baseline", run_baseline, "Qpast1-b-baseline"),
        ("hybrid", run_hybrid, "Qpast1-b-hybrid"),
    ]

    for m in M_VALUES:
        for model, run_model, algorithm in models:
            run_model(
                gameName=GAME_NAME,
                Nact=NACT,
                roundsG=ROUNDS,
                algoName=algorithm,
                lr=LR,
                Nagent=NAGENT,
                simStart=1,
                Nsim=NSIM,
                tStart=0,
                T=T,
                m=m,
                seed=SEED,
                tau=TAU,
            )

            print(f"Completed: {model}, m={m}")

    print("\nMain assortativity experiment completed.")


if __name__ == "__main__":
    main()