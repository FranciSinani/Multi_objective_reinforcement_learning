# Multi-Objective Reinforcement Learning - Deep Sea Treasure Concave

## Overview

This project implements and compares Multi-Objective Reinforcement Learning
(MORL) methods on the `deep-sea-treasure-concave-v0` benchmark from
MO-Gymnasium.

The agent controls a submarine in a grid world. The objectives are conflicting:

| Objective | Goal | Meaning |
| --- | --- | --- |
| Time cost | Minimize | Reaching deeper treasures takes more steps |
| Treasure value | Maximize | Deeper treasures usually have larger values |

The project uses maximization form for Pareto fronts and metrics:

```text
reported/plot form = (-time_cost, treasure)
```

So higher values are better on both axes.

## Implemented Methods

### Tabular scalarized Q-learning

The scalarized tabular methods share one implementation in
`tabular_scalarized_q_learning.py`. They differ only by the scalarization
function used to rank normalized objective Q-values.

Implemented tabular methods:

| Command | Method | Preference parameter |
| --- | --- | --- |
| `python .\main.py mo` | Weighted Sum Q-Learning | Weights |
| `python .\main.py owa` | OWA Q-Learning | OWA weights |
| `python .\main.py cheb` | Chebyshev Q-Learning | Chebyshev weights |
| `python .\main.py choquet` | Choquet Q-Learning | Choquet capacities |

### Deep scalarized Q-learning

The deep methods use a shared vector-valued DQN engine in
`deep_morl_common.py`. Method wrappers and experiment runners are in
`deep_scalarized_q_learning.py`.

Implemented deep methods:

| Command | Method | Preference parameter |
| --- | --- | --- |
| `py -3.13 .\main.py deep-ws` | Deep Weighted Sum Q-Learning | Weights |
| `py -3.13 .\main.py deep-owa` | Deep OWA Q-Learning | OWA weights |
| `py -3.13 .\main.py deep-cheb` | Deep Chebyshev Q-Learning | Chebyshev weights |
| `py -3.13 .\main.py deep-choquet` | Deep Choquet Q-Learning | Choquet capacities |

### Pareto Q-Learning

Pareto Q-Learning is implemented in `pareto_q_learning.py`.

Unlike scalarized methods, it does not train one policy per preference. It
stores sets of non-dominated value vectors per state-action pair and attempts
to approximate the full Pareto front in one run.

Run it with:

```powershell
python .\main.py pql
```

## Main Formulas

### Objective normalization

Before scalarization, objective Q-values are normalized:

```text
Qbar_i = clip((Q_i - L_i) / (U_i - L_i), 0, 1)
```

For this environment:

```text
time range     = [-19, 0]
treasure range = [0, 124]
```

Normalization is used for scalarized action ranking. Final reported solutions
remain raw environment outcomes.

### Weighted Sum

```text
S(Qbar) = w_time * Qbar_time + w_treasure * Qbar_treasure
```

### OWA

OWA sorts the normalized objective values first:

```text
S(Qbar) = w_1 * largest(Qbar) + w_2 * smallest(Qbar)
```

### Chebyshev

The normalized ideal point is `(1, 1)`:

```text
S(Qbar) = - max_i w_i * abs(1 - Qbar_i)
```

The code maximizes the negative distance, which is equivalent to minimizing
the weighted Chebyshev distance from the ideal point.

### Choquet integral

For two objectives and capacity `(mu_time, mu_treasure, mu_both)`:

```text
if Qbar_time <= Qbar_treasure:
    C_mu = Qbar_time * mu_both
           + (Qbar_treasure - Qbar_time) * mu_treasure
else:
    C_mu = Qbar_treasure * mu_both
           + (Qbar_time - Qbar_treasure) * mu_time
```

In this project `mu_both = 1.0`.

### Tabular Bellman update

The next action is selected by scalarization:

```text
a_star = argmax_a S(Qbar(s_next, a))
```

Then both objective Q-values are updated using the same selected action:

```text
Q_i(s, a) <- Q_i(s, a)
             + alpha * [r_i + gamma * Q_i(s_next, a_star) - Q_i(s, a)]
```

For terminal transitions:

```text
Q_i(s, a) <- Q_i(s, a) + alpha * [r_i - Q_i(s, a)]
```

### Deep DQN target

The online network selects the next action:

```text
a_next = argmax_a S(Qbar_theta(s_next, a))
```

The target network computes the future value:

```text
y_i = r_i + gamma * (1 - done) * Q_target_i(s_next, a_next)
```

The network is trained using Huber loss with objective scaling:

```text
loss = Huber(Q_theta_i / scale_i, y_i / scale_i)
```

where:

```text
scale_time = 19
scale_treasure = 124
```

The target network uses soft updates:

```text
theta_target <- tau * theta_online + (1 - tau) * theta_target
```

with `tau = 0.005`.

## Code Structure

```text
config.py
    Shared experiment settings: weights, capacities, timesteps, gamma,
    learning rates, epsilon values, and result folders.

env.py
    Loads the Deep Sea Treasure environment and returns the true Pareto front
    in maximization form: (-time_cost, treasure).

scalarization.py
    Normalization and scalarization formulas:
    Weighted Sum, OWA, Chebyshev, and Choquet.

tabular_scalarized_q_learning.py
    Shared tabular vector Q-learning loop for Weighted Sum, OWA, Chebyshev,
    and Choquet.

deep_morl_common.py
    Shared vector-valued DQN engine: neural network, replay buffer, target
    network, Bellman targets, loss, evaluation, and archive.

deep_scalarized_q_learning.py
    Deep method wrappers and runners for all deep scalarized methods.

pareto_q_learning.py
    Pareto Q-Learning with set-valued Q-updates.

utils.py
    Pareto extraction, hypervolume, IGD, epsilon indicator, EUM, and JSON
    result saving.

plots.py
    Pareto plots and metric plots.

show_metrics.py
    Prints metric summaries from saved result folders.

main.py
    Main command-line entry point.
```

## Pipeline

```text
config.py
    -> main.py
        -> tabular_scalarized_q_learning.py
        -> deep_scalarized_q_learning.py
        -> pareto_q_learning.py
            -> env.py
            -> scalarization.py
            -> training and evaluation
            -> utils.py
            -> plots.py
            -> results/
```

Scalarized method flow:

```text
state
    -> Q-values for all actions
    -> normalize Q-values
    -> scalarization score
    -> epsilon-greedy action selection
    -> environment step
    -> Bellman update
    -> evaluation
    -> final solution and plots
```

Deep method flow:

```text
state
    -> one-hot encoding
    -> neural network
    -> vector Q-values for all actions
    -> scalarized action selection
    -> replay buffer
    -> minibatch DQN update
    -> target network update
    -> checkpoint/evaluation
```

Pareto Q-Learning flow:

```text
state
    -> set of Q-vectors per action
    -> nondominated Bellman set backup
    -> Pareto front approximation
```

## How to Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run tabular methods:

```powershell
python .\main.py mo
python .\main.py owa
python .\main.py cheb
python .\main.py choquet
```

Run Pareto Q-Learning:

```powershell
python .\main.py pql
```

Run deep methods:

```powershell
py -3.13 .\main.py deep-ws
py -3.13 .\main.py deep-owa
py -3.13 .\main.py deep-cheb
py -3.13 .\main.py deep-choquet
```

Run all tabular scalarized methods plus Pareto Q-Learning:

```powershell
python .\main.py all
```

Run comparison mode:

```powershell
python .\main.py compare
```

Note: `all` and `compare` currently cover the tabular scalarized methods plus
Pareto Q-Learning. Deep methods are run separately with the `deep-*` commands.

## Results

Each method saves its outputs in a separate folder under `results/`.

Examples:

```text
results/tabular_weighted_sum/
results/tabular_owa/
results/tabular_chebyshev/
results/tabular_choquet/
results/pareto_q_learning/
results/deep_weighted_sum/
results/deep_owa/
results/deep_chebyshev/
results/deep_choquet/
```

Typical saved files include:

```text
*_pareto_front.png
*_hv.png
*_igd.png
*_epsilon.png
final_solutions.json
```

## Metrics

The project reports:

| Metric | Direction | Meaning |
| --- | --- | --- |
| HV | Higher is better | Dominated objective-space volume |
| IGD | Lower is better | Average distance from true front to learned front |
| Epsilon | Lower is better | Minimum additive shift needed to dominate the true front |
| Coverage | Higher is better | Number of true Pareto solutions recovered |
| EUM | Higher is better | Expected utility over a set of weight vectors |

Metrics are computed in maximization form:

```text
(-time_cost, treasure)
```

## Notes

- Scalarization is used for action selection and Bellman next-action selection.
- Rewards and final reported points remain raw environment outcomes.
- Normalization is used only to make objectives comparable during scalarized
  ranking.
- Deep methods use replay memory, a target network, soft target updates, and
  best-checkpoint selection.
- Pareto Q-Learning is not preference-specific; it stores nondominated value
  sets and approximates a full front in one run.
