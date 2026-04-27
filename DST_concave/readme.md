# Multi-Objective Reinforcement Learning — Deep Sea Treasure (Concave)

## Overview

This project implements and compares four multi-objective reinforcement learning (MORL) algorithms on the **Deep Sea Treasure Concave** environment from [MO-Gymnasium].

The agent controls a submarine navigating a grid. At each cell the agent can collect a treasure. The two conflicting objectives are:

| Objective | Goal | Notes |
|-----------|------|-------|
| **Treasure value** | Maximise | Larger treasures are deeper in the grid |
| **Time cost** | Minimise | Every step costs one time unit |

Because larger treasures require more steps to reach, there is no single "best" solution. Instead there is a **Pareto front** of trade-off solutions. The goal of each algorithm is to approximate this front as well as possible.

---

## Algorithms

### 1. MO Q-Learning (`mo_q_learning.py`)
Learns two separate Q-tables — one per objective — then combines them with a **weighted sum** during action selection:

```
Q[s,a] = w_time * Q1[s,a] + w_treasure * Q2[s,a]
```

Run 9 times with different weight vectors. Each run finds one point on the front.  
**Limitation:** weighted sum can only reach convex parts of the Pareto front.

---

### 2. OWA Q-Learning (`owa_q_learning.py`)
Uses **Ordered Weighted Averaging** — sorts the objective Q-values before applying weights, so the weight attaches to the rank rather than to a fixed objective:

```
OWA([v1, v2], [w1, w2]) = w1 * max(v1,v2) + w2 * min(v1,v2)
```

Run 9 times. Produces solutions that are more balanced than weighted sum.

---

### 3. Chebyshev Q-Learning (`chebyshev_q_learning.py`)
Uses **Chebyshev scalarisation** — minimises the maximum weighted deviation from an ideal point:

```
Cheb([v1,v2], w, z) = max_i( w_i * |v_i - z_i| )
```

The ideal point `z` is updated online. The L-shaped contours of this function can theoretically reach **any** point on the Pareto front, including concave regions that weighted sum misses.  
Run 9 times with different weight vectors.

---

### 4. Pareto Q-Learning (`pareto_q_learning.py`)
Keeps **sets of non-dominated value vectors** per state-action pair instead of scalar Q-values. No scalarisation is used during learning. A single run produces an approximation of the **full** Pareto front. Action selection uses hypervolume as the greedy criterion.

---

## Metrics

Three metrics are tracked during training and reported at the end:

| Metric | Formula | Direction | What it measures |
|--------|---------|-----------|-----------------|
| **HV** — Hypervolume | Area dominated by solution set, bounded by a reference point | Higher = better | Convergence + spread + coverage in one number |
| **IGD** — Inverted Generational Distance | Average distance from each true front point to its nearest learned point | Lower = better | How well the learned front covers the true front |
| **ε** — Additive Epsilon Indicator | Minimum shift needed so learned front dominates true front | Lower = better | Worst-case gap between learned and true front |

At the end of `compare` mode, **EUM** (Expected Utility Metric) is also reported:

| Metric | What it measures |
|--------|-----------------|
| **EUM** — Expected Utility | Average best linear utility across all weight vectors; higher = better |

---

## Output Files

### Per-algorithm (4 images per algorithm)

Running `python main.py mo` (or `owa`, `cheb`, `pql`) saves:

```
results/
├── mo_pareto_front.png    ← Learned front vs true front
├── mo_hv.png              ← HV over time  +  HV staircase in objective space
├── mo_igd.png             ← IGD over time  +  distance arrows in objective space
└── mo_epsilon.png         ← Epsilon over time  +  shift visualisation
```

Same structure for `owa_`, `cheb_`, `pql_`.

### Comparison (4 images)

Running `python main.py compare` saves:

```
results/
├── comparison_hv.png      ← HV over time, all 4 algorithms on one plot
├── comparison_hv_obj.png  ← HV staircase view, all 4 final fronts together
├── comparison_igd.png     ← IGD over time, all 4 algorithms on one plot
└── comparison_epsilon.png ← Epsilon over time, all 4 algorithms on one plot
```

> **Note:** The Pareto-front panel is intentionally not included in the comparison output. Each algorithm's front is already available in its own dedicated `*_pareto_front.png` image.

---

## File Structure

```
.
├── main.py                   Entry point — controls which algorithms run
├── utils.py                  HV, IGD, Epsilon, EUM, dominance helpers
├── env.py                    Loads the true Pareto front from MO-Gymnasium
├── plots.py                  All plotting — separated images per metric
├── mo_q_learning.py          MO Q-Learning training
├── owa_q_learning.py         OWA Q-Learning training
├── chebyshev_q_learning.py   Chebyshev Q-Learning training
├── pareto_q_learning.py      Pareto Q-Learning training
├── requirements.txt          Python dependencies
└── results/                  All saved plots (created automatically)
```

---

## How to Run

**Install dependencies (once):**
```bash
pip install -r requirements.txt
```

**Run one algorithm:**
```bash
python main.py mo      # MO Q-Learning
python main.py owa     # OWA Q-Learning
python main.py cheb    # Chebyshev Q-Learning
python main.py pql     # Pareto Q-Learning
```

**Run all algorithms with individual plots:**
```bash
python main.py all
```

**Run all algorithms with comparison plots and final metrics table:**
```bash
python main.py compare
```

---

## Expected Results

| Algorithm | HV | IGD | Epsilon | Notes |
|-----------|-----|-----|---------|-------|
| MO Q-Learning | Medium | High | High | Misses concave regions |
| OWA Q-Learning | Medium | Medium | Medium | Better balance than weighted sum |
| Chebyshev Q-Learning | Medium–High | Medium | Medium | Reaches some concave regions |
| Pareto Q-Learning | **Highest** | **Lowest** | **Lowest** | Full front in one run |

PQL is expected to achieve the best metrics because it learns the full Pareto front without any fixed preference. The scalarisation methods are limited by the number and diversity of weight settings used.

---

## Coordinate Convention

All algorithms internally use different representations. Everything is converted to **maximisation form** before plotting and metric computation:

```
Internal training form:  (time_cost, treasure)
Maximisation form:       (-time_cost, treasure)   [both axes: larger = better]
Plot axes:               x = -Time Cost,  y = Treasure Value
```

The HV reference point is `(-100, 0)` in maximisation form.