# implements the baseline reinforcement-learning model with assortative rematching
# trains pd behaviour using q-learning and records cooperation, strategies, switching and agent-level outcomes

import os
import sys
import numpy as np
from numba import njit


# map state labels to q-table rows

@njit
def state_to_index(state):
    if state == 0:
        return 0
    if state == 1:
        return 1
    if state == 100:
        return 2
    return 3


# select an action using the boltzmann policy

@njit
def select_action(Q, agent, state, tau):
    state_index = state_to_index(state)

    q0 = tau * Q[agent, state_index, 0]
    q1 = tau * Q[agent, state_index, 1]
    maximum = max(q0, q1)

    e0 = np.exp(q0 - maximum)
    e1 = np.exp(q1 - maximum)
    probability_action_0 = e0 / (e0 + e1)

    return 0 if np.random.random() < probability_action_0 else 1


# get the probability of selecting one action

@njit
def get_action_probability(Q, agent, state, action, tau):
    state_index = state_to_index(state)

    q0 = tau * Q[agent, state_index, 0]
    q1 = tau * Q[agent, state_index, 1]
    maximum = max(q0, q1)

    e0 = np.exp(q0 - maximum)
    e1 = np.exp(q1 - maximum)

    if action == 0:
        return e0 / (e0 + e1)
    return e1 / (e0 + e1)


# randomly pair the full population

@njit
def create_random_pairs(number_agents):
    agent_ids = np.arange(number_agents)
    np.random.shuffle(agent_ids)

    pairs = np.empty((number_agents // 2, 2), dtype=np.int64)

    for pair_index, index in enumerate(range(0, number_agents, 2)):
        pairs[pair_index, 0] = agent_ids[index]
        pairs[pair_index, 1] = agent_ids[index + 1]

    return pairs


# pair agents by their previous pd action

@njit
def create_assortative_pairs(number_agents, actions_pd):
    cooperators = np.empty(number_agents, dtype=np.int64)
    defectors = np.empty(number_agents, dtype=np.int64)

    number_cooperators = 0
    number_defectors = 0

    for agent in range(number_agents):
        if actions_pd[agent] == 0:
            cooperators[number_cooperators] = agent
            number_cooperators += 1
        else:
            defectors[number_defectors] = agent
            number_defectors += 1

    cooperative_agents = cooperators[:number_cooperators].copy()
    defective_agents = defectors[:number_defectors].copy()

    np.random.shuffle(cooperative_agents)
    np.random.shuffle(defective_agents)

    pairs = np.empty((number_agents // 2, 2), dtype=np.int64)
    leftovers = np.empty(number_agents, dtype=np.int64)

    pair_index = 0
    leftover_index = 0
    index = 0

    while index + 1 < number_cooperators:
        pairs[pair_index] = cooperative_agents[index:index + 2]
        pair_index += 1
        index += 2

    if index < number_cooperators:
        leftovers[leftover_index] = cooperative_agents[index]
        leftover_index += 1

    index = 0

    while index + 1 < number_defectors:
        pairs[pair_index] = defective_agents[index:index + 2]
        pair_index += 1
        index += 2

    if index < number_defectors:
        leftovers[leftover_index] = defective_agents[index]
        leftover_index += 1

    if leftover_index > 0:
        remaining_agents = leftovers[:leftover_index].copy()
        np.random.shuffle(remaining_agents)

        index = 0

        while index + 1 < leftover_index:
            pairs[pair_index] = remaining_agents[index:index + 2]
            pair_index += 1
            index += 2

    return pairs


# rematch all agents randomly or assortatively

@njit
def make_baseline_groups(number_agents, actions_pd, m):
    if np.random.random() < m:
        return create_assortative_pairs(number_agents, actions_pd)

    return create_random_pairs(number_agents)


# store one experience for q-learning

@njit
def store_experience(
    memory_states,
    memory_actions,
    memory_rewards,
    memory_lengths,
    agent,
    state,
    action,
    reward,
):
    memory_index = memory_lengths[agent]

    memory_states[agent, memory_index] = state
    memory_actions[agent, memory_index] = action
    memory_rewards[agent, memory_index] = reward
    memory_lengths[agent] += 1


# update q-values after one repeated game

@njit
def update_q_values(
    Q,
    memory_states,
    memory_actions,
    memory_rewards,
    memory_lengths,
    learning_rate,
    gamma,
):
    for agent in range(Q.shape[0]):
        running_return = 0.0

        for memory_index in range(memory_lengths[agent] - 1, -1, -1):
            running_return = (
                memory_rewards[agent, memory_index]
                + gamma * running_return
            )

            state_index = state_to_index(memory_states[agent, memory_index])
            action = memory_actions[agent, memory_index]

            Q[agent, state_index, action] = (
                (1.0 - learning_rate) * Q[agent, state_index, action]
                + learning_rate * running_return
            )

        memory_lengths[agent] = 0


# classify a learned strategy

@njit
def classify_strategy(Q, agent, cooperative_state, defective_state):
    cooperative_index = state_to_index(cooperative_state)
    defective_index = state_to_index(defective_state)

    cooperate_after_cooperation = (
        Q[agent, cooperative_index, 0]
        > Q[agent, cooperative_index, 1]
    )

    cooperate_after_defection = (
        Q[agent, defective_index, 0]
        > Q[agent, defective_index, 1]
    )

    if cooperate_after_cooperation and cooperate_after_defection:
        return 0
    if cooperate_after_cooperation and not cooperate_after_defection:
        return 1
    if not cooperate_after_cooperation and cooperate_after_defection:
        return 2

    return 3


# run one baseline simulation

@njit
def run_simulation(
    number_agents,
    rounds_per_game,
    training_horizon,
    learning_rate,
    m,
    tau,
    gamma,
    seed,
):
    np.random.seed(seed)

    Q = np.zeros((number_agents, 4, 2), dtype=np.float64)

    reward_player_1 = np.array(
        [[3, 0], [5, 1]],
        dtype=np.float64,
    )

    reward_player_2 = np.array(
        [[3, 5], [0, 1]],
        dtype=np.float64,
    )

    actions_pd = np.random.randint(0, 2, size=number_agents)
    groups = create_random_pairs(number_agents)

    maximum_memory = rounds_per_game + 5

    memory_states = np.zeros(
        (number_agents, maximum_memory),
        dtype=np.int64,
    )

    memory_actions = np.zeros(
        (number_agents, maximum_memory),
        dtype=np.int64,
    )

    memory_rewards = np.zeros(
        (number_agents, maximum_memory),
        dtype=np.float64,
    )

    memory_lengths = np.zeros(number_agents, dtype=np.int64)

    count_outcome_t = np.zeros((training_horizon, 4), dtype=np.int64)
    count_actual_outcome_t = np.zeros((training_horizon, 8), dtype=np.int64)
    count_switch_t = np.zeros((training_horizon, 2), dtype=np.int64)

    count_policy_switch_type_t = np.zeros(
        (training_horizon, 4),
        dtype=np.int64,
    )

    count_policy_pd_type_t = np.zeros(
        (training_horizon, 4),
        dtype=np.int64,
    )

    agent_count_switch_t = np.zeros(
        (training_horizon, number_agents),
        dtype=np.int64,
    )

    agent_count_defection_t = np.zeros(
        (training_horizon, number_agents),
        dtype=np.int64,
    )

    stat_policy_mean_t = np.zeros(
        (training_horizon + 1, 8),
        dtype=np.float64,
    )

    stat_q_mean_t = np.zeros(
        (training_horizon + 1, 8),
        dtype=np.float64,
    )

    stat_policy_variance_t = np.zeros(
        (training_horizon + 1, 8),
        dtype=np.float64,
    )

    stat_q_variance_t = np.zeros(
        (training_horizon + 1, 8),
        dtype=np.float64,
    )

    stat_policy_mean_t[0, :] = 0.5
    states = np.array([0, 1, 100, 101], dtype=np.int64)

    for training_step in range(1, training_horizon + 1):
        count_switch = np.zeros(2, dtype=np.int64)
        count_outcome = np.zeros(4, dtype=np.int64)
        count_actual_outcome = np.zeros(8, dtype=np.int64)

        agent_count_switch = np.zeros(number_agents, dtype=np.int64)
        agent_count_defection = np.zeros(number_agents, dtype=np.int64)

        for game_round in range(rounds_per_game):
            groups = make_baseline_groups(
                number_agents,
                actions_pd,
                m,
            )

            is_switch = np.ones(number_agents, dtype=np.int64)

            count_switch[1] += number_agents
            agent_count_switch += 1

            for group_index in range(groups.shape[0]):
                first_agent = groups[group_index, 0]
                second_agent = groups[group_index, 1]

                first_state = 100 + actions_pd[second_agent]
                second_state = 100 + actions_pd[first_agent]

                first_action = select_action(
                    Q,
                    first_agent,
                    first_state,
                    tau,
                )

                second_action = select_action(
                    Q,
                    second_agent,
                    second_state,
                    tau,
                )

                first_reward = reward_player_1[
                    first_action,
                    second_action,
                ]

                second_reward = reward_player_2[
                    first_action,
                    second_action,
                ]

                store_experience(
                    memory_states, memory_actions, memory_rewards, memory_lengths,
                    first_agent, first_state, first_action, first_reward,
                )

                store_experience(
                    memory_states, memory_actions, memory_rewards, memory_lengths,
                    second_agent, second_state, second_action, second_reward,
                )

                outcome_index = 2 * first_action + second_action
                count_outcome[outcome_index] += 1

                actual_outcome_index = (
                    4 * is_switch[first_agent]
                    + outcome_index
                )

                count_actual_outcome[actual_outcome_index] += 1

                agent_count_defection[first_agent] += first_action
                agent_count_defection[second_agent] += second_action

                actions_pd[first_agent] = first_action
                actions_pd[second_agent] = second_action

        update_q_values(
            Q,
            memory_states,
            memory_actions,
            memory_rewards,
            memory_lengths,
            learning_rate,
            gamma,
        )

        index = training_step - 1

        count_switch_t[index] = count_switch
        count_outcome_t[index] = count_outcome
        count_actual_outcome_t[index] = count_actual_outcome
        agent_count_switch_t[index] = agent_count_switch
        agent_count_defection_t[index] = agent_count_defection

        for agent in range(number_agents):
            switch_policy = classify_strategy(Q, agent, 0, 1)
            pd_policy = classify_strategy(Q, agent, 100, 101)

            count_policy_switch_type_t[index, switch_policy] += 1
            count_policy_pd_type_t[index, pd_policy] += 1

        q_sum = np.zeros(8, dtype=np.float64)
        policy_sum = np.zeros(8, dtype=np.float64)
        q_squared_sum = np.zeros(8, dtype=np.float64)
        policy_squared_sum = np.zeros(8, dtype=np.float64)

        for agent in range(number_agents):
            value_index = 0

            for state in states:
                state_index = state_to_index(state)

                for action in range(2):
                    q_value = Q[agent, state_index, action]

                    policy_probability = get_action_probability(
                        Q,
                        agent,
                        state,
                        action,
                        tau,
                    )

                    q_sum[value_index] += q_value
                    q_squared_sum[value_index] += q_value * q_value
                    policy_sum[value_index] += policy_probability
                    policy_squared_sum[value_index] += (
                        policy_probability * policy_probability
                    )

                    value_index += 1

        for value_index in range(8):
            stat_q_mean_t[training_step, value_index] = (
                q_sum[value_index] / number_agents
            )

            stat_q_variance_t[training_step, value_index] = (
                q_squared_sum[value_index] / number_agents
                - stat_q_mean_t[training_step, value_index] ** 2
            )

            stat_policy_mean_t[training_step, value_index] = (
                policy_sum[value_index] / number_agents
            )

            stat_policy_variance_t[training_step, value_index] = (
                policy_squared_sum[value_index] / number_agents
                - stat_policy_mean_t[training_step, value_index] ** 2
            )

    return (
        stat_q_mean_t,
        stat_q_variance_t,
        stat_policy_mean_t,
        stat_policy_variance_t,
        count_switch_t,
        count_outcome_t,
        count_actual_outcome_t,
        count_policy_switch_type_t,
        count_policy_pd_type_t,
        agent_count_switch_t,
        agent_count_defection_t,
        Q,
        groups,
        actions_pd,
    )


# save outputs from one simulation

def save_simulation_results(directory_name, simulation, output):
    (
        stat_q_mean_t,
        stat_q_variance_t,
        stat_policy_mean_t,
        stat_policy_variance_t,
        count_switch_t,
        count_outcome_t,
        count_actual_outcome_t,
        count_policy_switch_type_t,
        count_policy_pd_type_t,
        agent_count_switch_t,
        agent_count_defection_t,
        Q,
        groups,
        actions_pd,
    ) = output

    text_outputs = [
        ("StatQmeanT", stat_q_mean_t, "%.6f"),
        ("StatQvarT", stat_q_variance_t, "%.6f"),
        ("StatPmeanT", stat_policy_mean_t, "%.6f"),
        ("StatPvarT", stat_policy_variance_t, "%.6f"),
        ("CountSwitchT", count_switch_t, "%d"),
        ("CountOutcomeT", count_outcome_t, "%d"),
        ("CountActualOutcomeT", count_actual_outcome_t, "%d"),
        ("CountPolicySwTypeT", count_policy_switch_type_t, "%d"),
        ("CountPolicyPDTypeT", count_policy_pd_type_t, "%d"),
        ("AgentCountSwT", agent_count_switch_t, "%d"),
        ("AgentCountDT", agent_count_defection_t, "%d"),
    ]

    for name, data, fmt in text_outputs:
        np.savetxt(
            f"{directory_name}/{name}-sim{simulation:04d}.txt",
            data,
            fmt=fmt,
            delimiter=",",
        )

    np.save(f"{directory_name}/FinalQ-sim{simulation:04d}.npy", Q)
    np.save(f"{directory_name}/FinalGroups-sim{simulation:04d}.npy", groups)
    np.save(f"{directory_name}/FinalActionsPD-sim{simulation:04d}.npy", actions_pd)


# run a batch of independent simulations

def run_experiment_batch(
    gameName,
    Nact,
    roundsG,
    algoName,
    lr,
    Nagent,
    simStart,
    Nsim,
    tStart,
    T,
    m,
    seed=None,
    tau=1.0,
    gamma=1.0,
    output_root=None,
):
    if tStart != 0:
        raise NotImplementedError(
            "This Numba implementation currently requires tStart=0."
        )

    result_directory = (
        f"result_{gameName}_{algoName}_lr{lr:.2f}_Npast1"
        f"_Nagent{Nagent}_R{roundsG}_baseline_m{m:.2f}_gamma{gamma:.2f}"
    )

    if output_root is not None:
        result_directory = os.path.join(
            output_root,
            result_directory,
        )

    os.makedirs(result_directory, exist_ok=True)

    for simulation in range(simStart, Nsim + 1):
        simulation_directory = os.path.join(
            result_directory,
            f"sim{simulation:04d}",
        )

        os.makedirs(simulation_directory, exist_ok=True)

        simulation_seed = (
            1234 + simulation
            if seed is None
            else seed + simulation
        )

        output = run_simulation(
            Nagent,
            roundsG,
            T,
            lr,
            m,
            tau,
            gamma,
            simulation_seed,
        )

        save_simulation_results(
            simulation_directory,
            simulation,
            output,
        )

        print(
            f"Completed: baseline, sim={simulation}, "
            f"m={m:.2f}, gamma={gamma:.2f}"
        )


# default experiment settings

if __name__ == "__main__":
    number_simulations = 10
    training_horizon = 100000

    learning_rate = 0.05
    tau = 1.0
    gamma = 1.0

    number_agents = 20
    rounds_per_game = 20
    assortativity = 1.0

    if len(sys.argv) > 1:
        assortativity = float(sys.argv[1])
    if len(sys.argv) > 2:
        learning_rate = float(sys.argv[2])
    if len(sys.argv) > 3:
        tau = float(sys.argv[3])
    if len(sys.argv) > 4:
        number_simulations = int(sys.argv[4])
    if len(sys.argv) > 5:
        training_horizon = int(sys.argv[5])
    if len(sys.argv) > 6:
        gamma = float(sys.argv[6])
    if len(sys.argv) > 7:
        number_agents = int(sys.argv[7])

    run_experiment_batch(
        gameName="PD2",
        Nact=2,
        roundsG=rounds_per_game,
        algoName="Qpast1-b-baseline",
        lr=learning_rate,
        Nagent=number_agents,
        simStart=1,
        Nsim=number_simulations,
        tStart=0,
        T=training_horizon,
        m=assortativity,
        seed=1234,
        tau=tau,
        gamma=gamma,
    )