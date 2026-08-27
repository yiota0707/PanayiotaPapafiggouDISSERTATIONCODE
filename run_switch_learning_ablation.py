# runs the hybrid partner-switching learning ablation
# compares learned switching with switching learning disabled at the same experimental settings

import argparse
import os
import MAgame2106s2_hybrid_switch_ablation_njit as hybrid


# run one switch learning condition

def run_condition(
    condition_name,
    learn_switching,
    nsim,
    training_horizon,
    learning_rate,
    tau,
    number_agents,
    rounds_per_game,
    assortativity,
):
    output_root = os.path.join(
        "switch_learning_ablation_results",
        condition_name,
    )

    os.makedirs(output_root, exist_ok=True)

    hybrid.run_experiment_batch(
        gameName="PD2",
        Nact=2,
        roundsG=rounds_per_game,
        algoName="Qpast1-b-hybrid-switch-ablation",
        lr=learning_rate,
        Nagent=number_agents,
        simStart=1,
        Nsim=nsim,
        tStart=0,
        T=training_horizon,
        m=assortativity,
        seed=1234,
        tau=tau,
        gamma=1.0,
        learn_switching=learn_switching,
        output_root=output_root,
    )

    print(f"Completed: {condition_name}")


# run switch learning ablation

def main():
    parser = argparse.ArgumentParser(
        description="Run the Hybrid switch-learning ablation."
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
            "hybrid_learned_switching",
            "hybrid_no_switch_learning",
        ],
    )

    args = parser.parse_args()

    conditions = [
        ("hybrid_learned_switching", True),
        ("hybrid_no_switch_learning", False),
    ]

    if args.only is not None:
        conditions = [
            (name, learn_switching)
            for name, learn_switching in conditions
            if name in args.only
        ]

    for condition_name, learn_switching in conditions:
        run_condition(
            condition_name=condition_name,
            learn_switching=learn_switching,
            nsim=args.nsim,
            training_horizon=args.T,
            learning_rate=args.lr,
            tau=args.tau,
            number_agents=args.nagent,
            rounds_per_game=args.rounds,
            assortativity=args.m,
        )

    print("\nSwitch-learning ablation completed.")


if __name__ == "__main__":
    main()