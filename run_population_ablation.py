# runs the population-size sensitivity experiment for baseline and hybrid models
# compares cooperation across n=10, 20, 50 and 100 at m=0 and m=1

from MAgame2106s2_baseline_assort_m_njit import run_experiment_batch as run_baseline
from MAgame2106s2_hybrid_assort_m_njit import run_experiment_batch as run_hybrid


# experiment settings

NAGENT_VALUES = [10, 20, 50, 100]
M_VALUES = [0.0, 1.0]

GAME_NAME = "PD2"
NACT = 2
ROUNDS = 20
LR = 0.05
TAU = 1.0
NSIM = 10
T = 100000
SEED = 1234


# run population size sensitivity analysis

def main():
    models = [
        ("baseline", run_baseline, "Qpast1-b-baseline"),
        ("hybrid", run_hybrid, "Qpast1-b-hybrid"),
    ]

    for number_agents in NAGENT_VALUES:
        for m in M_VALUES:
            for model, run_model, algorithm in models:
                run_model(
                    gameName=GAME_NAME,
                    Nact=NACT,
                    roundsG=ROUNDS,
                    algoName=algorithm,
                    lr=LR,
                    Nagent=number_agents,
                    simStart=1,
                    Nsim=NSIM,
                    tStart=0,
                    T=T,
                    m=m,
                    seed=SEED,
                    tau=TAU,
                )

                print(f"Completed: {model}, N={number_agents}, m={m}")

    print("\nPopulation size sensitivity analysis completed.")


if __name__ == "__main__":
    main()