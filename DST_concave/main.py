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

Each training call returns 7 values:
    (final_point, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts)
"""

import os, sys
import numpy as np

from mo_q_learning        import train_mo_q
from owa_q_learning       import train_owa_q
from chebyshev_q_learning import train_chebyshev_q
from choquet_integral     import train_choquet_q
from pareto_q_learning    import train_pql
from env                  import get_true_reference_pf
from utils import (
    compute_hypervolume_2d,
    compute_igd,
    compute_epsilon_indicator,
    expected_utility_metric,
    dict_points_to_maximize,
    list_points_to_maximize,
    extract_pareto_front,
)
from plots import (
    plot_mo_q_results,
    plot_owa_q_results,
    plot_chebyshev_q_results,
    plot_choquet_q_results,
    plot_pql_results,
    plot_all_comparisons,
)

os.makedirs("results", exist_ok=True)

#weight settings (used by all scalarisation methods) 
WEIGHTS = [
    (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
    (0.6, 0.4), (0.5, 0.5), (0.4, 0.6),
    (0.3, 0.7), (0.2, 0.8), (0.1, 0.9),
]
CHOQUET_CAPACITIES = [
    # Synergy: mu1 + mu2 < mu12. Rewards solutions where both objectives are good.
    (0.10, 0.10, 1.0),
    (0.20, 0.20, 1.0),
    (0.30, 0.30, 1.0),
    (0.40, 0.40, 1.0),
    (0.15, 0.35, 1.0),
    (0.35, 0.15, 1.0),
    (0.20, 0.50, 1.0),
    (0.50, 0.20, 1.0),

    # Additive baselines: equivalent to weighted-sum behaviour.
    (0.25, 0.75, 1.0),
    (0.50, 0.50, 1.0),
    (0.75, 0.25, 1.0),

    # Redundancy: mu1 + mu2 > mu12. Limits compensation between objectives.
    (0.60, 0.60, 1.0),
    (0.70, 0.70, 1.0),
    (0.80, 0.80, 1.0),
    (0.90, 0.90, 1.0),
    (0.80, 0.40, 1.0),
    (0.40, 0.80, 1.0),
    (0.90, 0.60, 1.0),
    (0.60, 0.90, 1.0),
]
TIMESTEPS    = 400_000
LR           = 0.1
DEEP_CHOQUET_LR = 1e-3
GAMMA        = 0.99
EPS_START    = 1.0
EPS_END      = 0.05
LOG_INTERVAL = 1_000

MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "pql"


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
    mo_hv_ts,  mo_hv_pts = {}, {}
    mo_igd_ts, mo_igd_pts = {}, {}
    mo_eps_ts, mo_eps_pts = {}, {}

    for w in WEIGHTS:
        print(f"  weights={w}")
        pt, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts = train_mo_q(
            timeW=w[0], treasureW=w[1],
            total_timesteps=TIMESTEPS, lr=LR, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_end=EPS_END,
            log_interval=LOG_INTERVAL,
        )
        mo_ep[w]      = pt
        mo_hv_ts[w]   = hv_ts;  mo_hv_pts[w]  = hv_pts
        mo_igd_ts[w]  = igd_ts; mo_igd_pts[w] = igd_pts
        mo_eps_ts[w]  = eps_ts; mo_eps_pts[w] = eps_pts

    if MODE == "mo":
        plot_mo_q_results(
            mo_ep,
            mo_hv_ts,  mo_hv_pts,
            mo_igd_ts, mo_igd_pts,
            mo_eps_ts, mo_eps_pts,
            WEIGHTS,
        )


# OWA Q-Learning

if MODE in ("owa", "all", "compare"):
    print("\n[OWA Q-Learning]")
    owa_ep = {}
    owa_hv_ts,  owa_hv_pts = {}, {}
    owa_igd_ts, owa_igd_pts = {}, {}
    owa_eps_ts, owa_eps_pts = {}, {}

    for w in WEIGHTS:
        print(f"  weights={w}")
        pt, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts = train_owa_q(
            owa_weights=w,
            total_timesteps=TIMESTEPS, lr=LR, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_end=EPS_END,
            log_interval=LOG_INTERVAL,
        )
        owa_ep[w]      = pt
        owa_hv_ts[w]   = hv_ts;  owa_hv_pts[w]  = hv_pts
        owa_igd_ts[w]  = igd_ts; owa_igd_pts[w] = igd_pts
        owa_eps_ts[w]  = eps_ts; owa_eps_pts[w] = eps_pts

    if MODE == "owa":
        plot_owa_q_results(
            owa_ep,
            owa_hv_ts,  owa_hv_pts,
            owa_igd_ts, owa_igd_pts,
            owa_eps_ts, owa_eps_pts,
            WEIGHTS,
        )


# Chebyshev Q-Learning

if MODE in ("cheb", "all", "compare"):
    print("\n[Chebyshev Q-Learning]")
    cheb_ep = {}
    cheb_hv_ts,  cheb_hv_pts = {}, {}
    cheb_igd_ts, cheb_igd_pts = {}, {}
    cheb_eps_ts, cheb_eps_pts = {}, {}

    for w in WEIGHTS:
        print(f"  weights={w}")
        pt, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts = train_chebyshev_q(
            cheb_weights=w,
            total_timesteps=TIMESTEPS, lr=LR, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_end=EPS_END,
            log_interval=LOG_INTERVAL,
        )
        cheb_ep[w]      = pt
        cheb_hv_ts[w]   = hv_ts;  cheb_hv_pts[w]  = hv_pts
        cheb_igd_ts[w]  = igd_ts; cheb_igd_pts[w] = igd_pts
        cheb_eps_ts[w]  = eps_ts; cheb_eps_pts[w] = eps_pts

    if MODE == "cheb":
        plot_chebyshev_q_results(
            cheb_ep,
            cheb_hv_ts,  cheb_hv_pts,
            cheb_igd_ts, cheb_igd_pts,
            cheb_eps_ts, cheb_eps_pts,
            WEIGHTS,
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
    )

    if MODE == "pql":
        plot_pql_results(
            pql_ep,
            pql_hv_ts,  pql_hv_pts,
            pql_igd_ts, pql_igd_pts,
            pql_eps_ts, pql_eps_pts,
        )


# Choquet Integral Q-Learning

if MODE in ("choquet", "all", "compare"):
    print("\n[Choquet Integral Q-Learning]")
    choquet_ep = {}
    choquet_hv_ts, choquet_hv_pts = {}, {}
    choquet_igd_ts, choquet_igd_pts = {}, {}
    choquet_eps_ts, choquet_eps_pts = {}, {}

    for c in CHOQUET_CAPACITIES:
        print(f"  capacity={c}")
        pt, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts = train_choquet_q(
            mu1=c[0],
            mu2=c[1],
            mu12=c[2],
            total_timesteps=TIMESTEPS,
            lr=DEEP_CHOQUET_LR,
            # The reference Pareto front is generated with gamma=1.0.
            gamma=1.0,
            epsilon_start=EPS_START,
            epsilon_end=EPS_END,
            log_interval=LOG_INTERVAL,
        )
        choquet_ep[c]      = pt
        choquet_hv_ts[c]   = hv_ts;  choquet_hv_pts[c]  = hv_pts
        choquet_igd_ts[c]  = igd_ts; choquet_igd_pts[c] = igd_pts
        choquet_eps_ts[c]  = eps_ts; choquet_eps_pts[c] = eps_pts

    if MODE == "choquet":
        plot_choquet_q_results(
            choquet_ep,
            choquet_hv_ts,  choquet_hv_pts,
            choquet_igd_ts, choquet_igd_pts,
            choquet_eps_ts, choquet_eps_pts,
            CHOQUET_CAPACITIES,
        )


# All individual plots

if MODE == "all":
    plot_mo_q_results(
        mo_ep, mo_hv_ts, mo_hv_pts, mo_igd_ts, mo_igd_pts,
        mo_eps_ts, mo_eps_pts, WEIGHTS,
    )
    plot_owa_q_results(
        owa_ep, owa_hv_ts, owa_hv_pts, owa_igd_ts, owa_igd_pts,
        owa_eps_ts, owa_eps_pts, WEIGHTS,
    )
    plot_chebyshev_q_results(
        cheb_ep, cheb_hv_ts, cheb_hv_pts, cheb_igd_ts, cheb_igd_pts,
        cheb_eps_ts, cheb_eps_pts, WEIGHTS,
    )
    plot_choquet_q_results(
        choquet_ep, choquet_hv_ts, choquet_hv_pts,
        choquet_igd_ts, choquet_igd_pts,
        choquet_eps_ts, choquet_eps_pts, CHOQUET_CAPACITIES,
    )
    plot_pql_results(
        pql_ep, pql_hv_ts, pql_hv_pts, pql_igd_ts, pql_igd_pts,
        pql_eps_ts, pql_eps_pts,
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
