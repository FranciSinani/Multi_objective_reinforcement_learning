"""Shared experiment configuration for Deep Sea Treasure runs."""

RESULT_DIRS = {
    "mo": "results/tabular_weighted_sum",
    "owa": "results/tabular_owa",
    "cheb": "results/tabular_chebyshev",
    "choquet": "results/tabular_choquet",
    "pql": "results/pareto_q_learning",
    "compare": "results/comparisons",
    "deep_weighted_sum": "results/deep_weighted_sum",
    "deep_owa": "results/deep_owa",
    "deep_chebyshev": "results/deep_chebyshev",
    "deep_choquet": "results/deep_choquet",
}

WEIGHTS = [
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.40, 0.60),
    (0.35, 0.65),
    (0.30, 0.70),
    (0.25, 0.75),
    (0.20, 0.80),
    (0.15, 0.85),
    (0.10, 0.90),
    (0.05, 0.95),
]

ADDITIVE_CHOQUET_CAPACITIES = [
    (time_weight, treasure_weight, 1.0)
    for time_weight, treasure_weight in WEIGHTS
]

SYNERGY_CHOQUET_CAPACITIES = [
    (0.10, 0.10, 1.0),
    (0.20, 0.20, 1.0),
    (0.30, 0.30, 1.0),
    (0.40, 0.40, 1.0),
    (0.10, 0.30, 1.0),
    (0.30, 0.10, 1.0),
    (0.10, 0.50, 1.0),
    (0.50, 0.10, 1.0),
    (0.20, 0.50, 1.0),
    (0.50, 0.20, 1.0),
    (0.30, 0.50, 1.0),
    (0.50, 0.30, 1.0),
]

CHOQUET_CAPACITIES = (
    ADDITIVE_CHOQUET_CAPACITIES + SYNERGY_CHOQUET_CAPACITIES
)

TIMESTEPS = 200_000
TABULAR_LR = 0.1
DEEP_LR = 1e-3
GAMMA = 0.99
PQL_GAMMA = 1.0
EPSILON_START = 1.0
EPSILON_END = 0.05
LOG_INTERVAL = 1_000
OWA_EXPERIMENT_SEED = 8
