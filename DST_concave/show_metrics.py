"""Print final method-quality metrics from saved final solutions."""

import argparse
import json
import os

from env import get_true_reference_pf
from utils import (
    compute_epsilon_indicator,
    compute_hypervolume_2d,
    compute_igd,
)

METHOD_FILES = {
    "tabular-weighted-sum": (
        "results/tabular_weighted_sum/final_solutions.json"
    ),
    "tabular-owa": "results/tabular_owa/final_solutions.json",
    "tabular-chebyshev": (
        "results/tabular_chebyshev/final_solutions.json"
    ),
    "tabular-choquet": (
        "results/tabular_choquet/final_solutions.json"
    ),
    "pql": "results/pareto_q_learning/final_solutions.json",
    "deep-weighted-sum": (
        "results/deep_weighted_sum/final_solutions.json"
    ),
    "deep-owa": "results/deep_owa/final_solutions.json",
    "deep-chebyshev": (
        "results/deep_chebyshev/final_solutions.json"
    ),
    "deep-choquet": "results/deep_choquet/final_solutions.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        nargs="?",
        default="tabular-weighted-sum",
        choices=sorted(METHOD_FILES),
    )
    args = parser.parse_args()
    results_file = METHOD_FILES[args.method]

    if not os.path.exists(results_file):
        raise SystemExit(
            f"No saved metrics for '{args.method}'. "
            "Run that method first."
        )

    with open(results_file, encoding="utf-8") as file:
        result = json.load(file)

    method_front = [tuple(point) for point in result["final_solutions"]]
    true_front = get_true_reference_pf()
    coverage = sum(
        any(
            abs(true_point[0] - learned_point[0]) <= 0.5
            and abs(true_point[1] - learned_point[1]) <= 0.5
            for learned_point in method_front
        )
        for true_point in true_front
    )

    print(result["method"])
    print(f"HV: {compute_hypervolume_2d(method_front, (-100, 0)):.4f}")
    print(f"IGD: {compute_igd(true_front, method_front):.4f}")
    print(
        "Epsilon: "
        f"{compute_epsilon_indicator(true_front, method_front):.4f}"
    )
    print(f"Coverage: {coverage}/{len(true_front)}")


if __name__ == "__main__":
    main()
