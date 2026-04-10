import sys

from mo_q_learning import train_mo_q
from owa_q_learning import train_owa_q
from chebyshev_q_learning import train_chebyshev_q
from pareto_q_learning import train_pql
from plots import (
    plot_mo_q_results,
    plot_owa_q_results,
    plot_chebyshev_q_results,
    plot_pql_results,
    plot_all_comparisons,
)

RUN_MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "mo"

weights_list = [
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.5, 0.5),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.4, 0.6),
    (0.6, 0.4),
    (0.1, 0.9),
]

owa_settings = [
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.5, 0.5),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.4, 0.6),
    (0.6, 0.4),
    (0.1, 0.9),
]

cheb_settings = [
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.5, 0.5),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.4, 0.6),
    (0.6, 0.4),
    (0.1, 0.9),
]

# mo_q_learning

if RUN_MODE in ["mo", "all", "compare"]:
    all_episode_points = {}
    all_hv_timesteps = {}
    all_hv_points = {}

    for timeW, treasureW in weights_list:
        episode_points, hv_timesteps, hv_points = train_mo_q(
            timeW=timeW,
            treasureW=treasureW,
            total_timesteps=200000,
            lr=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            log_interval=1000,
        )

        all_episode_points[(timeW, treasureW)] = episode_points
        all_hv_timesteps[(timeW, treasureW)] = hv_timesteps
        all_hv_points[(timeW, treasureW)] = hv_points

    if RUN_MODE == "mo":
        plot_mo_q_results(
            all_episode_points,
            all_hv_timesteps,
            all_hv_points,
            weights_list,
        )


# owa_q_learning

if RUN_MODE in ["owa", "all", "compare"]:
    all_episode_points_owa = {}
    all_hv_timesteps_owa = {}
    all_hv_points_owa = {}

    for owa_w in owa_settings:
        episode_points, hv_timesteps, hv_points = train_owa_q(
            owa_weights=owa_w,
            total_timesteps=200000,
            lr=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            log_interval=1000,
        )

        all_episode_points_owa[owa_w] = episode_points
        all_hv_timesteps_owa[owa_w] = hv_timesteps
        all_hv_points_owa[owa_w] = hv_points

    if RUN_MODE == "owa":
        plot_owa_q_results(
            all_episode_points_owa,
            all_hv_timesteps_owa,
            all_hv_points_owa,
            owa_settings,
        )


# chebyshev_q_learning

if RUN_MODE in ["cheb", "all", "compare"]:
    all_episode_points_cheb = {}
    all_hv_timesteps_cheb = {}
    all_hv_points_cheb = {}

    for cheb_w in cheb_settings:
        episode_points, hv_timesteps, hv_points = train_chebyshev_q(
            cheb_weights=cheb_w,
            total_timesteps=200000,
            lr=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            log_interval=1000,
        )

        all_episode_points_cheb[cheb_w] = episode_points
        all_hv_timesteps_cheb[cheb_w] = hv_timesteps
        all_hv_points_cheb[cheb_w] = hv_points

    if RUN_MODE == "cheb":
        plot_chebyshev_q_results(
            all_episode_points_cheb,
            all_hv_timesteps_cheb,
            all_hv_points_cheb,
            cheb_settings,
        )



# pareto_q_learning

if RUN_MODE in ["pql", "all", "compare"]:
    episode_points_pql, hv_timesteps_pql, hv_points_pql = train_pql(
        total_timesteps=200000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        log_interval=1000,
    )

    if RUN_MODE == "pql":
        plot_pql_results(
            episode_points_pql,
            hv_timesteps_pql,
            hv_points_pql,
        )



# all individual plots

if RUN_MODE == "all":
    plot_mo_q_results(
        all_episode_points,
        all_hv_timesteps,
        all_hv_points,
        weights_list,
    )

    plot_owa_q_results(
        all_episode_points_owa,
        all_hv_timesteps_owa,
        all_hv_points_owa,
        owa_settings,
    )

    plot_chebyshev_q_results(
        all_episode_points_cheb,
        all_hv_timesteps_cheb,
        all_hv_points_cheb,
        cheb_settings,
    )

    plot_pql_results(
        episode_points_pql,
        hv_timesteps_pql,
        hv_points_pql,
    )



# combined comparison plot

if RUN_MODE == "compare":
    plot_all_comparisons(
        all_episode_points,
        all_episode_points_owa,
        all_episode_points_cheb,
        episode_points_pql,
        all_hv_timesteps,
        all_hv_points,
        all_hv_timesteps_owa,
        all_hv_points_owa,
        all_hv_timesteps_cheb,
        all_hv_points_cheb,
        hv_timesteps_pql,
        hv_points_pql,
        weights_list,
        owa_settings,
        cheb_settings,
    )