# Multi-Objective Reinforcement Learning on Deep Sea Treasure Concave

## Overview

In this project, I explored several multi-objective reinforcement learning approaches on the **Deep Sea Treasure Concave** environment from **MO-Gymnasium**.

My goal was to compare how different algorithms handle the trade-off between:

- **treasure value** (which I want to maximize)
- **time cost** (which I want to minimize)

To keep the comparison consistent, I represented the final solutions in objective space using:

- **x-axis = - time cost**
- **y-axis = treasure value**

This way, both objectives are treated in a maximization style when I plot Pareto fronts and compute hypervolume.

---

I implemented and compared four approaches:

### 1. MO Q-Learning
In this version, I learned two separate Q-value tables:

- one for the **time objective**
- one for the **treasure objective**

Then I combined them with a standard weighted sum during action selection. This let me test different preference settings, such as giving more importance to time or more importance to treasure.

---

### 2. OWA Q-Learning
In this version, I still learned separate Q-values for each objective, but instead of using a simple weighted sum, I used an **Ordered Weighted Averaging (OWA)** utility function. It does not attach weights directly to fixed objectives. Instead, it sorts the objective values first and then applies the weights. Because of that, I could make the algorithm care more about the better objective or the worse objective, depending on the chosen OWA setting.

---

### 3. Chebyshev Q-Learning
In this version, I used a **Chebyshev scalarization** method. Instead of averaging objectives, I made the algorithm focus on the **largest weighted deviation from an ideal point**. This helped me study a different kind of compromise-seeking behavior, especially when I wanted to avoid solutions that are too weak in one objective.

---

### 4. Pareto Q-Learning
This was the most directly multi-objective method. Instead of reducing the objectives to one scalar during learning, I kept **sets of non-dominated value vectors** for state-action pairs. Then I used those sets to build policies that approximate the Pareto front more directly. This method is especially useful when preferences of the decision maker or system are unknown.

---

## Environment

I used:

- **MO-Gymnasium**
- environment: **`deep-sea-treasure-concave-v0`**

This environment is a good benchmark for MORL because it contains a clear trade-off:

- small treasures can be reached quickly
- larger treasures require more time

The concave version is interesting because it highlights the limitations of scalarization methods on **non-convex Pareto fronts**.

---

## Evaluation

I evaluated the algorithms in two main ways.

### Pareto Front
I plotted the solutions in objective space using:

- **x = - time cost**
- **y = treasure value**

Then I extracted the non-dominated points to visualize the Pareto front approximation learned by each method.

### Hypervolume metric
I also computed **hypervolume over timesteps**. It gives a single metric that reflects both:

- the quality of the solutions
- the coverage of the Pareto front

This made it useful for comparing the algorithms over time.

---

## The comparison

The scalarization-based methods:

- MO Q-Learning
- OWA Q-Learning
- Chebyshev Q-Learning

all need some kind of preference during learning. For those methods, I ran multiple settings and then collected the discovered solutions into a set.
Pareto Q-Learning was different because it did not require fixed preferences during learning. It directly learned a set of trade-off solutions.
To compare them fairly, I compared the **solution sets they produced**, rather than trying to force them to use the same preference structure.

---

## File Structure

```text
.
├── main.py
├── utils.py
├── plots.py
├── mo_q_learning.py
├── owa_q_learning.py
├── chebyshev_q_learning.py
└── pareto_q_learning.py
```
---
## Running the project 

Install the requirements: 

- pip install -r requirements.txt

To run one algorithm only: 

- python main.py mo 
- python main.py owa 
- python main.py cheb 
- python main.py pql 

To run all algorithms: 

- python main.py all 

To run the combined comparison plot: 

- python main.py compare


