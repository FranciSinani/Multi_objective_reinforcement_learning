"""
Entry point for all MORL experiments on Deep Sea Treasure (concave).

Usage:
    python main.py mo        # MO Q-Learning only
    python main.py owa       # OWA Q-Learning only
    python main.py cheb      # Chebyshev Q-Learning only
    python main.py choquet   # Choquet Integral Q-Learning only
    python main.py pql       # Pareto Q-Learning only
    python main.py all       # all 5 algorithms + individual plots
    python main.py compare   # all 5 algorithms + comparison plots + metrics table

Archive-enabled scalarized training calls return:
    (final_point, archive_front,
     hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts)
"""

import os, sys
import numpy as np

MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "pql"

if MODE in ("mo", "all", "compare"):
    from mo_q_learning import train_mo_q
if MODE in ("owa", "all", "compare"):
    from owa_q_learning import train_owa_q
if MODE in ("cheb", "all", "compare"):
    from chebyshev_q_learning import train_chebyshev_q
if MODE in ("choquet", "all", "compare"):
    from choquet_tabular_q_learning import train_tabular_choquet_q
if MODE in ("pql", "all", "compare"):
    from pareto_q_learning import train_pql

from env                  import get_true_reference_pf
from utils import (
    compute_hypervolume_2d,
    compute_igd,
    compute_epsilon_indicator,
    expected_utility_metric,
    dict_points_to_maximize,
    list_points_to_maximize,
    extract_pareto_front,
    save_final_solutions,
)
from plots import (
    plot_mo_q_results,
    plot_owa_q_results,
    plot_chebyshev_q_results,
    plot_choquet_q_results,
    plot_pql_results,
    plot_all_comparisons,
)

RESULT_DIRS = {
    "mo": "results/tabular_weighted_sum",
    "owa": "results/tabular_owa",
    "cheb": "results/tabular_chebyshev",
    "choquet": "results/tabular_choquet",
    "pql": "results/pareto_q_learning",
    "compare": "results/comparisons",
}
for result_dir in RESULT_DIRS.values():
    os.makedirs(result_dir, exist_ok=True)

#weight settings (used by all scalarisation methods) 
WEIGHTS = [
    (0.90, 0.10), (0.80, 0.20), (0.70, 0.30),
    (0.60, 0.40), (0.50, 0.50), (0.40, 0.60),
    (0.35, 0.65), (0.30, 0.70), (0.25, 0.75),
    (0.20, 0.80), (0.15, 0.85), (0.10, 0.90),
    (0.05, 0.95),
]
CHOQUET_CAPACITIES = [
    # Additive capacities: same preference grid as the weight-based methods.
    (0.90, 0.10, 1.0),
    (0.80, 0.20, 1.0),
    (0.70, 0.30, 1.0),
    (0.60, 0.40, 1.0),
    (0.50, 0.50, 1.0),
    (0.40, 0.60, 1.0),
    (0.35, 0.65, 1.0),
    (0.30, 0.70, 1.0),
    (0.25, 0.75, 1.0),
    (0.20, 0.80, 1.0),
    (0.15, 0.85, 1.0),
    (0.10, 0.90, 1.0),
    (0.05, 0.95, 1.0),

    # Synergy: mu1 + mu2 < mu12. Rewards solutions where both objectives are good.
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
TIMESTEPS    = 200_000
WEIGHTED_SUM_TIMESTEPS = 200_000
LR           = 0.1
GAMMA        = 0.99
EPS_START    = 1.0
EPS_END      = 0.05
LOG_INTERVAL = 1_000
OWA_EXPERIMENT_SEED = 8

def count_true_front_coverage(true_front, learned_front, tol=0.5):
    """
    Count how many true Pareto-front points are exactly recovered.

    Both inputs must be in maximisation form: (-time_cost, treasure).
    """
    if not true_front or not learned_front:
        return 0
    count = 0
    for true_pt in true_front:
        if any(np.linalg.norm(np.array(true_pt) - np.array(pt)) <= tol
               for pt in learned_front):
            count += 1
    return count

# MO Q-Learning

if MODE in ("mo", "all", "compare"):
    print("\n[MO Q-Learning]")
    mo_ep = {}
    mo_final = {}
    mo_hv_ts,  mo_hv_pts = {}, {}
    mo_igd_ts, mo_igd_pts = {}, {}
    mo_eps_ts, mo_eps_pts = {}, {}

    for w in WEIGHTS:
        print(f"  weights={w}")
        pt, archive_front, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts = train_mo_q(
            timeW=w[0], treasureW=w[1],
            total_timesteps=WEIGHTED_SUM_TIMESTEPS, lr=LR, gamma=0.99,
            epsilon_start=EPS_START, epsilon_end=EPS_END,
            log_interval=LOG_INTERVAL,
            seed=7 + WEIGHTS.index(w),
        )
        mo_final[w]   = pt
        mo_ep[w]      = archive_front
        mo_hv_ts[w]   = hv_ts;  mo_hv_pts[w]  = hv_pts
        mo_igd_ts[w]  = igd_ts; mo_igd_pts[w] = igd_pts
        mo_eps_ts[w]  = eps_ts; mo_eps_pts[w] = eps_pts
        print(f"  final point: {pt}")

    if MODE == "mo":
        save_final_solutions(
            RESULT_DIRS["mo"],
            "Tabular Weighted-Sum Q-Learning",
            extract_pareto_front(dict_points_to_maximize(mo_final)),
            mo_final,
        )
        plot_mo_q_results(
            mo_ep,
            mo_hv_ts,  mo_hv_pts,
            mo_igd_ts, mo_igd_pts,
            mo_eps_ts, mo_eps_pts,
            WEIGHTS,
            mo_final,
            RESULT_DIRS["mo"],
        )


# OWA Q-Learning

if MODE in ("owa", "all", "compare"):
    print("\n[OWA Q-Learning]")
    owa_ep = {}
    owa_final = {}
    owa_hv_ts,  owa_hv_pts = {}, {}
    owa_igd_ts, owa_igd_pts = {}, {}
    owa_eps_ts, owa_eps_pts = {}, {}

    for w in WEIGHTS:
        print(f"  weights={w}")
        pt, archive_front, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts = train_owa_q(
            owa_weights=w,
            total_timesteps=TIMESTEPS, lr=LR, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_end=EPS_END,
            log_interval=LOG_INTERVAL,
            seed=OWA_EXPERIMENT_SEED * 100 + WEIGHTS.index(w),
        )
        owa_final[w]   = pt
        owa_ep[w]      = archive_front
        owa_hv_ts[w]   = hv_ts;  owa_hv_pts[w]  = hv_pts
        owa_igd_ts[w]  = igd_ts; owa_igd_pts[w] = igd_pts
        owa_eps_ts[w]  = eps_ts; owa_eps_pts[w] = eps_pts
        print(f"  final point: {pt}")

    if MODE == "owa":
        save_final_solutions(
            RESULT_DIRS["owa"],
            "Tabular OWA Q-Learning",
            extract_pareto_front(dict_points_to_maximize(owa_final)),
            owa_final,
        )
        plot_owa_q_results(
            owa_ep,
            owa_hv_ts,  owa_hv_pts,
            owa_igd_ts, owa_igd_pts,
            owa_eps_ts, owa_eps_pts,
            WEIGHTS,
            owa_final,
            RESULT_DIRS["owa"],
        )


# Chebyshev Q-Learning

if MODE in ("cheb", "all", "compare"):
    print("\n[Chebyshev Q-Learning]")
    cheb_ep = {}
    cheb_final = {}
    cheb_hv_ts,  cheb_hv_pts = {}, {}
    cheb_igd_ts, cheb_igd_pts = {}, {}
    cheb_eps_ts, cheb_eps_pts = {}, {}

    for w in WEIGHTS:
        print(f"  weights={w}")
        pt, archive_front, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts = train_chebyshev_q(
            cheb_weights=w,
            total_timesteps=TIMESTEPS, lr=LR, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_end=EPS_END,
            log_interval=LOG_INTERVAL,
            seed=7 + WEIGHTS.index(w),
        )
        cheb_final[w]   = pt
        cheb_ep[w]      = archive_front
        cheb_hv_ts[w]   = hv_ts;  cheb_hv_pts[w]  = hv_pts
        cheb_igd_ts[w]  = igd_ts; cheb_igd_pts[w] = igd_pts
        cheb_eps_ts[w]  = eps_ts; cheb_eps_pts[w] = eps_pts
        print(f"  final point: {pt}")

    if MODE == "cheb":
        save_final_solutions(
            RESULT_DIRS["cheb"],
            "Tabular Chebyshev Q-Learning",
            extract_pareto_front(dict_points_to_maximize(cheb_final)),
            cheb_final,
        )
        plot_chebyshev_q_results(
            cheb_ep,
            cheb_hv_ts,  cheb_hv_pts,
            cheb_igd_ts, cheb_igd_pts,
            cheb_eps_ts, cheb_eps_pts,
            WEIGHTS,
            cheb_final,
            RESULT_DIRS["cheb"],
        )


# Pareto Q-Learning

if MODE in ("pql", "all", "compare"):
    print("\n[Pareto Q-Learning]")
    pql_ep, pql_hv_ts, pql_hv_pts, pql_igd_ts, pql_igd_pts, \
        pql_eps_ts, pql_eps_pts = train_pql(
        total_timesteps=TIMESTEPS,
        gamma=1.0,
        epsilon_start=EPS_START,
        epsilon_end=EPS_END,
        log_interval=LOG_INTERVAL,
        seed=7,
    )

    if MODE == "pql":
        save_final_solutions(
            RESULT_DIRS["pql"],
            "Pareto Q-Learning",
            extract_pareto_front(list_points_to_maximize(pql_ep)),
        )
        plot_pql_results(
            pql_ep,
            pql_hv_ts,  pql_hv_pts,
            pql_igd_ts, pql_igd_pts,
            pql_eps_ts, pql_eps_pts,
            RESULT_DIRS["pql"],
        )


# Choquet Integral Q-Learning

if MODE in ("choquet", "all", "compare"):
    print("\n[Choquet Integral Q-Learning]")
    choquet_ep = {}
    choquet_final = {}
    choquet_hv_ts, choquet_hv_pts = {}, {}
    choquet_igd_ts, choquet_igd_pts = {}, {}
    choquet_eps_ts, choquet_eps_pts = {}, {}

    for c in CHOQUET_CAPACITIES:
        print(f"  capacity={c}")
        pt, archive_front, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts = train_tabular_choquet_q(
            mu1=c[0],
            mu2=c[1],
            mu12=c[2],
            total_timesteps=TIMESTEPS,
            lr=LR,
            gamma=GAMMA,
            epsilon_start=EPS_START,
            epsilon_end=EPS_END,
            log_interval=LOG_INTERVAL,
            seed=7 + CHOQUET_CAPACITIES.index(c),
        )
        choquet_final[c]   = pt
        choquet_ep[c]      = archive_front
        choquet_hv_ts[c]   = hv_ts;  choquet_hv_pts[c]  = hv_pts
        choquet_igd_ts[c]  = igd_ts; choquet_igd_pts[c] = igd_pts
        choquet_eps_ts[c]  = eps_ts; choquet_eps_pts[c] = eps_pts
        print(f"  final point: {pt}")

    if MODE == "choquet":
        save_final_solutions(
            RESULT_DIRS["choquet"],
            "Tabular Choquet Q-Learning",
            extract_pareto_front(dict_points_to_maximize(choquet_final)),
            choquet_final,
            preference_label="capacity",
        )
        plot_choquet_q_results(
            choquet_ep,
            choquet_hv_ts,  choquet_hv_pts,
            choquet_igd_ts, choquet_igd_pts,
            choquet_eps_ts, choquet_eps_pts,
            CHOQUET_CAPACITIES,
            choquet_final,
            RESULT_DIRS["choquet"],
        )


# All individual plots

if MODE == "all":
    save_final_solutions(
        RESULT_DIRS["mo"],
        "Tabular Weighted-Sum Q-Learning",
        extract_pareto_front(dict_points_to_maximize(mo_final)),
        mo_final,
    )
    save_final_solutions(
        RESULT_DIRS["owa"],
        "Tabular OWA Q-Learning",
        extract_pareto_front(dict_points_to_maximize(owa_final)),
        owa_final,
    )
    save_final_solutions(
        RESULT_DIRS["cheb"],
        "Tabular Chebyshev Q-Learning",
        extract_pareto_front(dict_points_to_maximize(cheb_final)),
        cheb_final,
    )
    save_final_solutions(
        RESULT_DIRS["choquet"],
        "Tabular Choquet Q-Learning",
        extract_pareto_front(dict_points_to_maximize(choquet_final)),
        choquet_final,
        preference_label="capacity",
    )
    save_final_solutions(
        RESULT_DIRS["pql"],
        "Pareto Q-Learning",
        extract_pareto_front(list_points_to_maximize(pql_ep)),
    )
    plot_mo_q_results(
        mo_ep, mo_hv_ts, mo_hv_pts, mo_igd_ts, mo_igd_pts,
        mo_eps_ts, mo_eps_pts, WEIGHTS, mo_final, RESULT_DIRS["mo"],
    )
    plot_owa_q_results(
        owa_ep, owa_hv_ts, owa_hv_pts, owa_igd_ts, owa_igd_pts,
        owa_eps_ts, owa_eps_pts, WEIGHTS, owa_final, RESULT_DIRS["owa"],
    )
    plot_chebyshev_q_results(
        cheb_ep, cheb_hv_ts, cheb_hv_pts, cheb_igd_ts, cheb_igd_pts,
        cheb_eps_ts, cheb_eps_pts, WEIGHTS, cheb_final,
        RESULT_DIRS["cheb"],
    )
    plot_choquet_q_results(
        choquet_ep, choquet_hv_ts, choquet_hv_pts,
        choquet_igd_ts, choquet_igd_pts,
        choquet_eps_ts, choquet_eps_pts, CHOQUET_CAPACITIES,
        choquet_final,
        RESULT_DIRS["choquet"],
    )
    plot_pql_results(
        pql_ep, pql_hv_ts, pql_hv_pts, pql_igd_ts, pql_igd_pts,
        pql_eps_ts, pql_eps_pts, RESULT_DIRS["pql"],
    )

# Comparison plots + final metrics table

if MODE == "compare":
    plot_all_comparisons(
        mo_ep,   owa_ep,   cheb_ep,   choquet_ep,   pql_ep,
        mo_hv_ts,    mo_hv_pts,
        owa_hv_ts,   owa_hv_pts,
        cheb_hv_ts,  cheb_hv_pts,
        choquet_hv_ts, choquet_hv_pts,
        pql_hv_ts,   pql_hv_pts,
        mo_igd_ts,   mo_igd_pts,
        owa_igd_ts,  owa_igd_pts,
        cheb_igd_ts, cheb_igd_pts,
        choquet_igd_ts, choquet_igd_pts,
        pql_igd_ts,  pql_igd_pts,
        mo_eps_ts,   mo_eps_pts,
        owa_eps_ts,  owa_eps_pts,
        cheb_eps_ts, cheb_eps_pts,
        choquet_eps_ts, choquet_eps_pts,
        pql_eps_ts,  pql_eps_pts,
        WEIGHTS, WEIGHTS, WEIGHTS, CHOQUET_CAPACITIES,
        RESULT_DIRS["compare"],
    )

    # final metrics on non-dominated solution sets 
    mo_max   = dict_points_to_maximize(mo_ep)
    owa_max  = dict_points_to_maximize(owa_ep)
    cheb_max = dict_points_to_maximize(cheb_ep)
    choquet_max = dict_points_to_maximize(choquet_ep)
    pql_max  = list_points_to_maximize(pql_ep)

    mo_front   = extract_pareto_front(mo_max)
    owa_front  = extract_pareto_front(owa_max)
    cheb_front = extract_pareto_front(cheb_max)
    choquet_front = extract_pareto_front(choquet_max)
    pql_front  = extract_pareto_front(pql_max)

    true_pf = get_true_reference_pf()
    REF     = (-100, 0)

    print("\n" + "=" * 68)
    print("  FINAL METRICS  (on non-dominated solution sets)")
    print("=" * 68)
    print(f"  {'Algorithm':<30}  {'HV':>10}  {'IGD':>10}  {'EPS':>10}  {'EUM':>10}  {'Covered':>10}")
    print("  " + "-" * 78)
    for name, front in [
        ("MO Q-Learning",        mo_front),
        ("OWA Q-Learning",       owa_front),
        ("Chebyshev Q-Learning", cheb_front),
        ("Choquet Integral Q-Learning", choquet_front),
        ("Pareto Q-Learning",    pql_front),
    ]:
        hv  = compute_hypervolume_2d(front, ref_point=REF)
        igd = compute_igd(true_pf, front)
        eps = compute_epsilon_indicator(true_pf, front)
        eum = expected_utility_metric(front, WEIGHTS)
        covered = count_true_front_coverage(true_pf, front)
        coverage = f"{covered}/{len(true_pf)}"
        print(f"  {name:<30}  {hv:>10.4f}  {igd:>10.4f}  {eps:>10.4f}  {eum:>10.4f}  {coverage:>10}")
    print("=" * 68)
    print("  HV, EUM, Covered : higher = better  |  IGD, EPS : lower = better")
    print("=" * 68)
