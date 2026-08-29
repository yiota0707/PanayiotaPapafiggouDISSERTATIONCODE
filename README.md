# MSc Dissertation Code

This directory contains the Python code used for the computational experiments and analyses reported in the MSc dissertation:

**Assortative Matching and the Emergence of Cooperation in Multi-Agent Reinforcement Learning Systems**

The project investigates cooperation in repeated Prisoner's Dilemma interactions between reinforcement-learning agents under assortative rematching. Two models are compared:

- **Baseline:** agents learn their Prisoner's Dilemma behaviour using Q-learning, while partner rematching is determined by the assortative matching mechanism. Partner stay/switch decisions are not learned.
- **Hybrid:** agents learn both their Prisoner's Dilemma behaviour and partner stay/switch decisions using Q-learning, alongside assortative rematching.

The experiments use repeated independent simulations to investigate the effects of assortativity, model parameters and selected model components on cooperation.

---

## Core Model Files

The core model files implement the reinforcement-learning environments used by the experiment runners. These files are imported by the corresponding `run_*.py` scripts and do not normally need to be executed directly.

### `MAgame2106s2_baseline_assort_m_njit.py`

Implements the main Baseline reinforcement-learning model with assortative rematching. Agents learn their Prisoner's Dilemma behaviour using Q-learning, while partner stay/switch decisions are not learned.

### `MAgame2106s2_hybrid_assort_m_njit.py`

Implements the main Hybrid model. Agents learn both their Prisoner's Dilemma behaviour and partner stay/switch decisions using Q-learning, alongside assortative rematching.

### `MAgame2106s2_baseline_ablation_njit.py`

Baseline model implementation used for the future-return learning ablation.

### `MAgame2106s2_hybrid_ablation_njit.py`

Hybrid model implementation used for the future-return learning ablation.

### `MAgame2106s2_hybrid_switch_ablation_njit.py`

Hybrid model implementation used to compare learned partner switching with the condition in which switching learning is disabled.

---

## Main Experiment Scripts

### `run_main_m_ablation.py`

Runs the main assortativity experiment for the Baseline and Hybrid models across the tested values of the assortativity parameter \(m\).

### `run_lr_ablation.py`

Runs the learning-rate sensitivity experiment across multiple learning rates at \(m=0\) and \(m=1\).

### `run_population_ablation.py`

Runs the population-size sensitivity experiment for populations of 10, 20, 50 and 100 agents.

### `run_tau_ablation.py`

Runs the temperature sensitivity experiment across the tested softmax temperature values.

### `run_horizon_ablation.py`

Runs the training-horizon sensitivity experiment at 50,000, 100,000 and 200,000 training iterations.

---

## Component and Additional Experiments

### `run_component_ablations.py`

Runs the future-return learning ablation for both models by comparing the full configuration with the condition in which future-return learning is removed.

### `run_switch_learning_ablation.py`

Runs the Hybrid partner-switching ablation by comparing learned switching with the condition in which switching learning is disabled.

### `run_initial_cooperation_bias_experiment.py`

Tests whether cooperation-biased initial Q-values affect final cooperation, residual defection and learned strategies.

### `run_cooperation_bonus_ablation.py`

Tests how an additional cooperation reward affects cooperation, residual defection and learned strategies.

---

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

Analyses the future-return learning ablation and compares cooperation and learned behaviour across conditions.

### `analyse_switch_learning_ablation.py`

Analyses the partner-switching learning ablation, including cooperation, switching behaviour and policy composition.

---

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

```text
m = {0.00, 0.30, 0.50, 0.70, 1.00}
```

Parameter-sensitivity experiments generally compare the Baseline and Hybrid models at \(m=0\) and \(m=1\).

---

## Installation and Setup

The code requires Python 3.

Install the required Python packages using:

```bash
pip install numpy pandas scipy matplotlib numba
```

The main dependencies are:

- NumPy
- pandas
- SciPy
- Matplotlib
- Numba

After installing the dependencies, place the Python scripts in the same project directory.

The experiment scripts can then be executed from a terminal opened in that directory.

Some experiments are computationally intensive, particularly those involving larger population sizes, longer training horizons or repeated independent simulations. Numba JIT compilation is used to reduce simulation runtime.

---

## Running the Code

The core model files are imported by the experiment scripts and do not normally need to be executed directly.

The intended workflow is:

1. Run the required experiment script to generate simulation results.
2. Allow the experiment to produce its corresponding result directories.
3. Run the relevant analysis scripts after the required experimental results have been generated.

### 1. Main Assortativity Experiment

Run the main Baseline-Hybrid assortativity experiment using:

```bash
python run_main_m_ablation.py
```

This executes the Baseline and Hybrid models across:

```text
m = {0.00, 0.30, 0.50, 0.70, 1.00}
```

with repeated independent simulations for each condition.

### 2. Parameter-Sensitivity Experiments

The parameter-sensitivity experiments can be run using:

```bash
python run_lr_ablation.py
python run_population_ablation.py
python run_tau_ablation.py
python run_horizon_ablation.py
```

These evaluate learning rate, population size, softmax temperature and training horizon, respectively.

### 3. Component Ablations and Additional Experiments

The component and additional experiments can be run using:

```bash
python run_component_ablations.py
python run_switch_learning_ablation.py
python run_initial_cooperation_bias_experiment.py
python run_cooperation_bonus_ablation.py
```

`run_component_ablations.py` evaluates the removal of future-return learning by comparing the full models with the γ condition.

`run_switch_learning_ablation.py` evaluates the effect of disabling learned partner switching in the Hybrid model.

`run_initial_cooperation_bias_experiment.py` evaluates the effect of cooperation-biased Q-value initialisation.

`run_cooperation_bonus_ablation.py` evaluates the effect of adding an explicit cooperation bonus during learning.

### 4. Analysis

Analysis scripts should be executed only after the result directories required by the corresponding analysis have been generated.

The principal analysis scripts are:

```bash
python analyse_all_experiments.py
python analyse_statistical_significance.py
python analyse_component_ablations.py
python analyse_switch_learning_ablation.py
python analyse_learning_speed.py
python analyse_behavioural_dynamics.py
python analyse_population_defector_hypothesis.py
```

The analysis scripts aggregate independent simulation outputs, calculate summary statistics and statistical comparisons, and generate the corresponding figures and behavioural analyses.

The overall workflow is:

```text
Core model files
       |
       v
Experiment runner scripts
       |
       v
Simulation result directories
       |
       v
Analysis scripts
       |
       v
Summary statistics and figures
```

The complete experimental programme is computationally intensive. In particular, experiments involving larger populations, longer training horizons and repeated simulations may require substantial execution time.

---

## Recommended Execution Order

For reproducing the complete set of experiments, the experiment runners can be executed in the following order:

```bash
python run_main_m_ablation.py

python run_lr_ablation.py
python run_population_ablation.py
python run_tau_ablation.py
python run_horizon_ablation.py

python run_component_ablations.py
python run_switch_learning_ablation.py
python run_initial_cooperation_bias_experiment.py
python run_cooperation_bonus_ablation.py
```

The analysis scripts can then be executed after their required result directories have been generated:

```bash
python analyse_all_experiments.py
python analyse_statistical_significance.py
python analyse_component_ablations.py
python analyse_switch_learning_ablation.py
python analyse_learning_speed.py
python analyse_behavioural_dynamics.py
python analyse_population_defector_hypothesis.py
```

The individual experiment groups are independent unless an analysis script requires outputs from more than one experiment. It is therefore not necessary to run every experiment when reproducing only a particular analysis.

---

## Software

The implementation is written in Python and uses:

- NumPy for numerical computation
- pandas for data processing and aggregation
- SciPy for statistical analysis
- Matplotlib for visualisation
- Numba for JIT compilation and improved simulation performance

---

## Reproducibility

Experimental conditions are evaluated using multiple independent simulations.

The experiment scripts define the parameter values used for each reported sensitivity or ablation study, while the analysis scripts process the resulting simulation outputs and calculate the summary statistics and statistical comparisons reported in the dissertation.

For full reproduction, the relevant experiment scripts should be executed before their corresponding analysis scripts. Because the complete experimental programme includes multiple conditions and repeated simulations, reproducing all reported results may require substantial computation time.

The accompanying dissertation provides the full model definitions, experimental methodology, interpretation of the results and discussion of limitations.
