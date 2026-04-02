import matplotlib.pyplot as plt
from utils import extract_pareto_front


def plot_mo_q_results(all_episode_points, all_hv_timesteps, all_hv_points, weights_list):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT: PARETO FRONT
    all_points = []
    for points in all_episode_points.values():
        for time_cost, treasure in points:
            all_points.append((-time_cost, treasure))

    pareto_front = extract_pareto_front(all_points)

    x_all = [p[0] for p in all_points]
    y_all = [p[1] for p in all_points]
    x_pf = [p[0] for p in pareto_front]
    y_pf = [p[1] for p in pareto_front]

    axes[0].scatter(x_all, y_all, alpha=0.15, label="All solutions")
    axes[0].scatter(x_pf, y_pf, s=60, label="Pareto front")
    axes[0].set_xlabel("- Time Cost")
    axes[0].set_ylabel("Treasure Value")
    axes[0].set_title("Pareto Front for MO Q-Learning")
    axes[0].grid(True)
    axes[0].legend()

    # RIGHT: HYPERVOLUME VS TIMESTEP
    for weights in weights_list:
        axes[1].plot(
            all_hv_timesteps[weights],
            all_hv_points[weights],
            marker="o",
            label=str(weights)
        )

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Hypervolume")
    axes[1].set_title("Hypervolume vs Timestep for MO Q-Learning")
    axes[1].grid(True)
    axes[1].legend(title="Weights")

    plt.tight_layout()
    plt.show()


def plot_owa_q_results(all_episode_points_owa, all_hv_timesteps_owa, all_hv_points_owa, owa_settings):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT: PARETO FRONT
    all_points = []
    for points in all_episode_points_owa.values():
        for time_cost, treasure in points:
            all_points.append((-time_cost, treasure))

    pareto_front = extract_pareto_front(all_points)

    x_all = [p[0] for p in all_points]
    y_all = [p[1] for p in all_points]
    x_pf = [p[0] for p in pareto_front]
    y_pf = [p[1] for p in pareto_front]

    axes[0].scatter(x_all, y_all, alpha=0.15, label="All solutions")
    axes[0].scatter(x_pf, y_pf, s=60, label="Pareto front")
    axes[0].set_xlabel("- Time Cost")
    axes[0].set_ylabel("Treasure Value")
    axes[0].set_title("Pareto Front for OWA Q-Learning")
    axes[0].grid(True)
    axes[0].legend()

    # RIGHT: HYPERVOLUME VS TIMESTEP
    for owa_w in owa_settings:
        axes[1].plot(
            all_hv_timesteps_owa[owa_w],
            all_hv_points_owa[owa_w],
            marker="o",
            label=str(owa_w)
        )

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Hypervolume")
    axes[1].set_title("Hypervolume vs Timestep for OWA Q-Learning")
    axes[1].grid(True)
    axes[1].legend(title="OWA weights")

    plt.tight_layout()
    plt.show()


def plot_chebyshev_q_results(all_episode_points_cheb, all_hv_timesteps_cheb, all_hv_points_cheb, cheb_settings):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT: PARETO FRONT
    all_points = []
    for points in all_episode_points_cheb.values():
        for time_cost, treasure in points:
            all_points.append((-time_cost, treasure))

    pareto_front = extract_pareto_front(all_points)

    x_all = [p[0] for p in all_points]
    y_all = [p[1] for p in all_points]
    x_pf = [p[0] for p in pareto_front]
    y_pf = [p[1] for p in pareto_front]

    axes[0].scatter(x_all, y_all, alpha=0.15, label="All solutions")
    axes[0].scatter(x_pf, y_pf, s=60, label="Pareto front")
    axes[0].set_xlabel("- Time Cost")
    axes[0].set_ylabel("Treasure Value")
    axes[0].set_title("Pareto Front for Chebyshev Q-Learning")
    axes[0].grid(True)
    axes[0].legend()

    # RIGHT: HYPERVOLUME VS TIMESTEP
    for cheb_w in cheb_settings:
        axes[1].plot(
            all_hv_timesteps_cheb[cheb_w],
            all_hv_points_cheb[cheb_w],
            marker="o",
            label=str(cheb_w)
        )

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Hypervolume")
    axes[1].set_title("Hypervolume vs Timestep for Chebyshev Q-Learning")
    axes[1].grid(True)
    axes[1].legend(title="Chebyshev weights")

    plt.tight_layout()
    plt.show()


def plot_pql_results(episode_points_pql, hv_timesteps_pql, hv_points_pql):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT: PARETO FRONT
    all_points = [(-time_cost, treasure) for time_cost, treasure in episode_points_pql]
    pareto_front = extract_pareto_front(all_points)

    x_all = [p[0] for p in all_points]
    y_all = [p[1] for p in all_points]
    x_pf = [p[0] for p in pareto_front]
    y_pf = [p[1] for p in pareto_front]

    axes[0].scatter(x_all, y_all, alpha=0.15, label="All solutions")
    axes[0].scatter(x_pf, y_pf, s=60, label="Pareto front")
    axes[0].set_xlabel("- Time Cost")
    axes[0].set_ylabel("Treasure Value")
    axes[0].set_title("Pareto Front for Pareto Q-Learning")
    axes[0].grid(True)
    axes[0].legend()

    # RIGHT: HYPERVOLUME VS TIMESTEP
    axes[1].plot(hv_timesteps_pql, hv_points_pql, marker="o", label="PQL")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Hypervolume")
    axes[1].set_title("Hypervolume vs Timestep for Pareto Q-Learning")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_all_comparisons(
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
):
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT: combined Pareto fronts
    all_points_mo = [(-tc, tr) for pts in all_episode_points.values() for tc, tr in pts]
    all_points_owa = [(-tc, tr) for pts in all_episode_points_owa.values() for tc, tr in pts]
    all_points_cheb = [(-tc, tr) for pts in all_episode_points_cheb.values() for tc, tr in pts]
    all_points_pql = [(-tc, tr) for tc, tr in episode_points_pql]

    pf_mo = extract_pareto_front(all_points_mo)
    pf_owa = extract_pareto_front(all_points_owa)
    pf_cheb = extract_pareto_front(all_points_cheb)
    pf_pql = extract_pareto_front(all_points_pql)

    axes[0].scatter([p[0] for p in pf_mo], [p[1] for p in pf_mo], s=55, label="MO Q-learning")
    axes[0].scatter([p[0] for p in pf_owa], [p[1] for p in pf_owa], s=55, label="OWA Q-learning")
    axes[0].scatter([p[0] for p in pf_cheb], [p[1] for p in pf_cheb], s=55, label="Chebyshev Q-learning")
    axes[0].scatter([p[0] for p in pf_pql], [p[1] for p in pf_pql], s=55, label="Pareto Q-learning")

    axes[0].set_xlabel("- Time Cost")
    axes[0].set_ylabel("Treasure Value")
    axes[0].set_title("Pareto Front Comparison")
    axes[0].grid(True)
    axes[0].legend()

    # RIGHT: combined hypervolume curves
    mo_hv_avg = np.mean(np.array([all_hv_points[w] for w in weights_list]), axis=0)
    mo_t = all_hv_timesteps[weights_list[0]]

    owa_hv_avg = np.mean(np.array([all_hv_points_owa[w] for w in owa_settings]), axis=0)
    owa_t = all_hv_timesteps_owa[owa_settings[0]]

    cheb_hv_avg = np.mean(np.array([all_hv_points_cheb[w] for w in cheb_settings]), axis=0)
    cheb_t = all_hv_timesteps_cheb[cheb_settings[0]]

    axes[1].plot(mo_t, mo_hv_avg, marker="o", label="MO Q-learning")
    axes[1].plot(owa_t, owa_hv_avg, marker="o", label="OWA Q-learning")
    axes[1].plot(cheb_t, cheb_hv_avg, marker="o", label="Chebyshev Q-learning")
    axes[1].plot(hv_timesteps_pql, hv_points_pql, marker="o", label="Pareto Q-learning")

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Hypervolume")
    axes[1].set_title("Hypervolume Comparison")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()