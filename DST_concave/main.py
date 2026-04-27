"""
main.py
=======
Entry point for all MORL experiments on Deep Sea Treasure (concave).

Usage
-----
    python main.py mo        # MO Q-Learning only
    python main.py owa       # OWA Q-Learning only
    python main.py cheb      # Chebyshev Q-Learning only
    python main.py pql       # Pareto Q-Learning only
    python main.py all       # all 4 algorithms + individual plots
    python main.py compare   # all 4 algorithms + comparison plots + metrics table

Each training call returns 7 values:
    (final_point, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts)
"""

import os, sys
import numpy as np

from mo_q_learning        import train_mo_q
from owa_q_learning       import train_owa_q
from chebyshev_q_learning import train_chebyshev_q
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
    plot_pql_results,
    plot_all_comparisons,
)

os.makedirs("results", exist_ok=True)

# shared weight settings (used by all scalarisation methods) 
WEIGHTS = [
    (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
    (0.6, 0.4), (0.5, 0.5), (0.4, 0.6),
    (0.3, 0.7), (0.2, 0.8), (0.1, 0.9),
]

TIMESTEPS    = 400_000
LR           = 0.1
GAMMA        = 0.99
EPS_START    = 1.0
EPS_END      = 0.05
LOG_INTERVAL = 1_000

MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "pql"

# =============================================================================
# MO Q-Learning
# =============================================================================
if MODE in ("mo", "all", "compare"):
    print("\n[MO Q-Learning]")
    mo_ep                                    = {}
    mo_hv_ts,  mo_hv_pts                    = {}, {}
    mo_igd_ts, mo_igd_pts                   = {}, {}
    mo_eps_ts, mo_eps_pts                   = {}, {}

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

# =============================================================================
# OWA Q-Learning
# =============================================================================
if MODE in ("owa", "all", "compare"):
    print("\n[OWA Q-Learning]")
    owa_ep                                    = {}
    owa_hv_ts,  owa_hv_pts                   = {}, {}
    owa_igd_ts, owa_igd_pts                  = {}, {}
    owa_eps_ts, owa_eps_pts                  = {}, {}

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

# =============================================================================
# Chebyshev Q-Learning
# =============================================================================
if MODE in ("cheb", "all", "compare"):
    print("\n[Chebyshev Q-Learning]")
    cheb_ep                                    = {}
    cheb_hv_ts,  cheb_hv_pts                  = {}, {}
    cheb_igd_ts, cheb_igd_pts                 = {}, {}
    cheb_eps_ts, cheb_eps_pts                 = {}, {}

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

# =============================================================================
# Pareto Q-Learning
# =============================================================================
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

# =============================================================================
# All individual plots
# =============================================================================
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
    plot_pql_results(
        pql_ep, pql_hv_ts, pql_hv_pts, pql_igd_ts, pql_igd_pts,
        pql_eps_ts, pql_eps_pts,
    )

# =============================================================================
# Comparison plots + final metrics table
# =============================================================================
if MODE == "compare":
    plot_all_comparisons(
        mo_ep,   owa_ep,   cheb_ep,   pql_ep,
        mo_hv_ts,    mo_hv_pts,
        owa_hv_ts,   owa_hv_pts,
        cheb_hv_ts,  cheb_hv_pts,
        pql_hv_ts,   pql_hv_pts,
        mo_igd_ts,   mo_igd_pts,
        owa_igd_ts,  owa_igd_pts,
        cheb_igd_ts, cheb_igd_pts,
        pql_igd_ts,  pql_igd_pts,
        mo_eps_ts,   mo_eps_pts,
        owa_eps_ts,  owa_eps_pts,
        cheb_eps_ts, cheb_eps_pts,
        pql_eps_ts,  pql_eps_pts,
        WEIGHTS, WEIGHTS, WEIGHTS,
    )

    # final metrics on non-dominated solution sets 
    mo_max   = dict_points_to_maximize(mo_ep)
    owa_max  = dict_points_to_maximize(owa_ep)
    cheb_max = dict_points_to_maximize(cheb_ep)
    pql_max  = list_points_to_maximize(pql_ep)

    mo_front   = extract_pareto_front(mo_max)
    owa_front  = extract_pareto_front(owa_max)
    cheb_front = extract_pareto_front(cheb_max)
    pql_front  = extract_pareto_front(pql_max)

    true_pf = get_true_reference_pf()
    REF     = (-100, 0)

    print("\n" + "=" * 68)
    print("  FINAL METRICS  (on non-dominated solution sets)")
    print("=" * 68)
    print(f"  {'Algorithm':<26}  {'HV':>10}  {'IGD':>10}  {'EPS':>10}  {'EUM':>10}")
    print("  " + "-" * 62)
    for name, front in [
        ("MO Q-Learning",        mo_front),
        ("OWA Q-Learning",       owa_front),
        ("Chebyshev Q-Learning", cheb_front),
        ("Pareto Q-Learning",    pql_front),
    ]:
        hv  = compute_hypervolume_2d(front, ref_point=REF)
        igd = compute_igd(true_pf, front)
        eps = compute_epsilon_indicator(true_pf, front)
        eum = expected_utility_metric(front, WEIGHTS)
        print(f"  {name:<26}  {hv:>10.4f}  {igd:>10.4f}  {eps:>10.4f}  {eum:>10.4f}")
    print("=" * 68)
    print("  HV, EUM : higher = better  |  IGD, EPS : lower = better")
    print("=" * 68)