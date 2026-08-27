# runs the discounting component ablation for both baseline and hybrid models
# compares the full models with gamma=1 against conditions where discounting is removed with gamma=0

import argparse
import os
import MAgame2106s2_baseline_ablation_njit as baseline
import MAgame2106s2_hybrid_ablation_njit as hybrid


# run one component ablation condition

def run_condition(
    condition_name,
    model_name,
    m,
    gamma,
    number_simulations,
    training_horizon,
    learning_rate,
    tau,
    number_agents,
    rounds_per_game,
):
    output_root = os.path.join(
        "component_ablation_results",
        condition_name,
    )
    os.makedirs(output_root, exist_ok=True)

    common_arguments = {
        "gameName": "PD2",
        "Nact": 2,
        "roundsG": rounds_per_game,
        "lr": learning_rate,
        "Nagent": number_agents,
        "simStart": 1,
        "Nsim": number_simulations,
        "tStart": 0,
        "T": training_horizon,
        "m": m,
        "seed": 1234,
        "tau": tau,
        "gamma": gamma,
        "output_root": output_root,
    }

    if model_name == "baseline":
        baseline.run_experiment_batch(
            algoName="Qpast1-b-baseline-ablation",
            **common_arguments,
        )
    elif model_name == "hybrid":
        hybrid.run_experiment_batch(
            algoName="Qpast1-b-hybrid-ablation",
            **common_arguments,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"Completed: {condition_name}")


# run component ablation study

def main():
    parser = argparse.ArgumentParser(
        description="Run discounting ablations for baseline and hybrid models."
    )

    parser.add_argument("--nsim", type=int, default=10)
    parser.add_argument("--T", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--nagent", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--m", type=float, default=1.0)

    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=[
            "baseline_full",
            "baseline_no_discounting",
            "hybrid_full",
            "hybrid_no_discounting",
        ],
    )

    args = parser.parse_args()

    conditions = [
        ("baseline_full", "baseline", 1.0),
        ("baseline_no_discounting", "baseline", 0.0),
        ("hybrid_full", "hybrid", 1.0),
        ("hybrid_no_discounting", "hybrid", 0.0),
    ]

    if args.only is not None:
        conditions = [
            condition
            for condition in conditions
            if condition[0] in args.only
        ]

    for condition_name, model_name, gamma in conditions:
        run_condition(
            condition_name=condition_name,
            model_name=model_name,
            m=args.m,
            gamma=gamma,
            number_simulations=args.nsim,
            training_horizon=args.T,
            learning_rate=args.lr,
            tau=args.tau,
            number_agents=args.nagent,
            rounds_per_game=args.rounds,
        )

    print("\nComponent ablation study completed.")


if __name__ == "__main__":
    main()