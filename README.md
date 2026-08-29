# MSc Dissertation Code

This directory contains the Python code used for the computational experiments and analyses reported in the MSc dissertation.

The project investigates cooperation in repeated Prisoner's Dilemma interactions between reinforcement-learning agents under assortative rematching. Two models are compared:

- **Baseline:** agents learn their Prisoner's Dilemma behaviour using Q-learning, while partner rematching is determined by the assortative matching mechanism. Partner stay/switch decisions are not learned.
- **Hybrid:** agents learn both their Prisoner's Dilemma behaviour and partner stay/switch decisions using Q-learning, alongside assortative rematching.

The experiments use repeated independent simulations and investigate the effect of assortativity, model parameters, and selected model components on cooperation.

## Core Model Files

### `MAgame2106s2_baseline_assort_m_njit.py`
Implements the main Baseline reinforcement-learning model with assortative rematching. Agents learn Prisoner's Dilemma behaviour using Q-learning.

### `MAgame2106s2_hybrid_assort_m_njit.py`
Implements the main Hybrid model. Agents learn both Prisoner's Dilemma behaviour and partner stay/switch decisions using Q-learning.

### `MAgame2106s2_baseline_ablation_njit.py`
Baseline model implementation used for the discounting component ablation.

### `MAgame2106s2_hybrid_ablation_njit.py`
Hybrid model implementation used for the discounting component ablation.

### `MAgame2106s2_hybrid_switch_ablation_njit.py`
Hybrid model implementation used to compare learned partner switching with switching learning disabled.

## Main Experiment Scripts

### `run_main_m_ablation.py`
Runs the main assortativity experiment for the Baseline and Hybrid models across different values of assortativity m.

### `run_lr_ablation.py`
Runs the learning-rate sensitivity experiment across multiple learning rates at m=0 and m=1.

### `run_population_ablation.py`
Runs the population-size sensitivity experiment for populations of 10, 20, 50 and 100 agents.

### `run_tau_ablation.py`
Runs the temperature sensitivity experiment across the tested softmax temperature values.

### `run_horizon_ablation.py`
Runs the training-horizon sensitivity experiment at 50,000, 100,000 and 200,000 training iterations.

## Component and Additional Experiments

### `run_component_ablations.py`
Runs the discounting component ablation for both models by comparing the full configuration with the condition in which discounting is removed.

### `run_switch_learning_ablation.py`
Runs the Hybrid partner-switching ablation by comparing learned switching with switching learning disabled.

### `run_initial_cooperation_bias_experiment.py`
Tests whether cooperation-biased initial Q-values affect final cooperation, residual defection and learned strategies.

### `run_cooperation_bonus_ablation.py`
Tests how an additional cooperation reward affects cooperation, residual defection and learned strategies.

## Analysis Scripts

### `analyse_all_experiments.py`
Collects and summarises results from the main assortativity and parameter-sensitivity experiments.

### `analyse_statistical_significance.py`
Performs statistical comparisons between the Baseline and Hybrid models, including 95% confidence intervals, Welch's t-tests, Mann-Whitney U tests and Cohen's d effect sizes.

### `analyse_learning_speed.py`
Analyses cooperation trajectories and compares learning and convergence behaviour between the Baseline and Hybrid models.

### `analyse_behavioural_dynamics.py`
Analyses behavioural dynamics including cooperation, switching, behavioural stability, strategy evolution and correlations with cooperation.

### `analyse_population_defector_hypothesis.py`
Analyses the relationship between population size, cooperation and residual defection, including persistent defectors and defector pairings.

### `analyse_component_ablations.py`
Analyses the discounting component ablation and compares cooperation and learned behaviour across conditions.

### `analyse_switch_learning_ablation.py`
Analyses the partner-switching learning ablation, including cooperation, switching behaviour and policy composition.

## Default Experimental Configuration

Unless varied by a sensitivity experiment, the main experimental configuration is:

- Prisoner's Dilemma: PD2
- Agents: 20
- Rounds per repeated game: 20
- Learning rate: 0.05
- Softmax temperature: 1.0
- Training horizon: 100,000 iterations
- Independent simulations per condition: 10
- Random seed: 1234

The main assortativity experiment evaluates:

m = {0.00, 0.30, 0.50, 0.70, 1.00}

Parameter-sensitivity experiments generally compare the Baseline and Hybrid models at m=0 and m=1.

## Installation and Setup

The code requires Python 3.

Install the required Python packages using:

    pip install numpy pandas scipy matplotlib numba

The main dependencies are:

- NumPy
- pandas
- SciPy
- Matplotlib
- Numba

After installing the dependencies, place the Python scripts in the same
project directory.

Experiments can then be run using the corresponding `run_*.py` scripts.

Some experiments are computationally intensive, particularly those using
larger population sizes or longer training horizons. Numba JIT compilation
is used to reduce simulation runtime.

## Running the Code

The main experiment can be run using:

    python run_main_m_ablation.py

Other experiments can be run using their corresponding `run_*.py` scripts.

Analysis scripts should be run after the required experimental output has been generated. For example:

    python analyse_all_experiments.py
    python analyse_statistical_significance.py

Some analysis scripts expect the result directories produced by their corresponding experiment scripts to be present.

## Software

The implementation is written in Python and uses NumPy, pandas, SciPy, Matplotlib and Numba. Numba is used to improve the computational efficiency of the reinforcement-learning simulations.

## Reproducibility

Experimental conditions are evaluated using multiple independent simulations. The experiment scripts define the parameter values used for each reported sensitivity or ablation study, while the analysis scripts calculate the summary statistics and statistical comparisons reported in the dissertation.

The accompanying dissertation provides the full model definitions, experimental methodology, interpretation of the results and discussion of limitations.
