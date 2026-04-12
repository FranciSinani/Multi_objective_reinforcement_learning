import matplotlib.pyplot as plt
import numpy as np
from utils import extract_pareto_front


def _points_from_dict_points(points_dict):
    """
    Accepts dict values in either of these forms:
        weights -> (time_cost, treasure)
    or
        weights -> [(time_cost, treasure)]
    or
        weights -> [(time_cost, treasure), ...]
    Converts them to maximization form: (-time_cost, treasure)
    """
    points = []

    for value in points_dict.values():
        if value is None:
            continue

        # Case 1: single point tuple like (time_cost, treasure)
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], (int, float, np.integer, np.floating))
            and isinstance(value[1], (int, float, np.integer, np.floating))
        ):
            time_cost, treasure = value
            points.append((-time_cost, treasure))

        # Case 2: list/tuple of points like [(time_cost, treasure), ...]
        elif isinstance(value, (list, tuple)):
            for item in value:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                ):
                    time_cost, treasure = item
                    points.append((-time_cost, treasure))

    return points


def _points_from_list_points(points_list):
    """
    Accepts:
        [(time_cost, treasure), ...]
    Converts to:
        [(-time_cost, treasure), ...]
    """
    points = []
    for item in points_list:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            time_cost, treasure = item
            points.append((-time_cost, treasure))
    return points


def _unique_points(points):
    return sorted(set(points), key=lambda p: (p[0], p[1]))


def _pareto_unique(points):
    return _unique_points(extract_pareto_front(points))


def _split_xy(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return x, y


def _plot_points_and_front(ax, all_points, title, all_label="All final points", front_label="Pareto front"):
    pareto_front = _pareto_unique(all_points)

    if all_points:
        x_all, y_all = _split_xy(all_points)
        ax.scatter(x_all, y_all, s=80, alpha=0.6, label=all_label)

    if pareto_front:
        x_pf, y_pf = _split_xy(pareto_front)
        ax.scatter(x_pf, y_pf, s=140, marker="x", linewidths=2.5, label=front_label, zorder=5)
        if len(pareto_front) > 1:
            ax.plot(x_pf, y_pf, linewidth=2, alpha=0.9)

    ax.set_xlabel("- Time Cost")
    ax.set_ylabel("Treasure Value")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def plot_mo_q_results(all_episode_points, all_hv_timesteps, all_hv_points, weights_list):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    all_points = _points_from_dict_points(all_episode_points)
    _plot_points_and_front(
        axes[0],
        all_points,
        "MO Q-Learning: Final Frozen Policy Returns"
    )

    for weights in weights_list:
        if weights in all_hv_timesteps and weights in all_hv_points:
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
    plt.savefig("results/mo_q_learning.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_owa_q_results(all_episode_points_owa, all_hv_timesteps_owa, all_hv_points_owa, owa_settings):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    all_points = _points_from_dict_points(all_episode_points_owa)
    _plot_points_and_front(
        axes[0],
        all_points,
        "OWA Q-Learning: Final Frozen Policy Returns"
    )

    for owa_w in owa_settings:
        if owa_w in all_hv_timesteps_owa and owa_w in all_hv_points_owa:
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
    plt.savefig("results/owa_q_learning.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_chebyshev_q_results(all_episode_points_cheb, all_hv_timesteps_cheb, all_hv_points_cheb, cheb_settings):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    all_points = _points_from_dict_points(all_episode_points_cheb)
    _plot_points_and_front(
        axes[0],
        all_points,
        "Chebyshev Q-Learning: Final Frozen Policy Returns"
    )

    for cheb_w in cheb_settings:
        if cheb_w in all_hv_timesteps_cheb and cheb_w in all_hv_points_cheb:
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
    plt.savefig("results/chebyshev_q_learning.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_pql_results(episode_points_pql, hv_timesteps_pql, hv_points_pql):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    all_points = _points_from_list_points(episode_points_pql)
    _plot_points_and_front(
        axes[0],
        all_points,
        "Pareto Q-Learning: Final Policy Returns"
    )

    axes[1].plot(hv_timesteps_pql, hv_points_pql, marker="o", label="PQL")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Hypervolume")
    axes[1].set_title("Hypervolume vs Timestep for Pareto Q-Learning")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/pql.png", dpi=300, bbox_inches="tight")
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    all_points_mo = _points_from_dict_points(all_episode_points)
    all_points_owa = _points_from_dict_points(all_episode_points_owa)
    all_points_cheb = _points_from_dict_points(all_episode_points_cheb)
    all_points_pql = _points_from_list_points(episode_points_pql)

    pf_mo = _pareto_unique(all_points_mo)
    pf_owa = _pareto_unique(all_points_owa)
    pf_cheb = _pareto_unique(all_points_cheb)
    pf_pql = _pareto_unique(all_points_pql)

    if all_points_mo:
        x_mo_all, y_mo_all = _split_xy(all_points_mo)
        axes[0].scatter(x_mo_all, y_mo_all, s=55, alpha=0.35, marker="o")

    if all_points_owa:
        x_owa_all, y_owa_all = _split_xy(all_points_owa)
        axes[0].scatter(x_owa_all, y_owa_all, s=55, alpha=0.35, marker="s")

    if all_points_cheb:
        x_cheb_all, y_cheb_all = _split_xy(all_points_cheb)
        axes[0].scatter(x_cheb_all, y_cheb_all, s=55, alpha=0.35, marker="x")

    if all_points_pql:
        x_pql_all, y_pql_all = _split_xy(all_points_pql)
        axes[0].scatter(x_pql_all, y_pql_all, s=55, alpha=0.35, marker="D")

    if pf_mo:
        x_mo, y_mo = _split_xy(pf_mo)
        axes[0].scatter(x_mo, y_mo, s=90, marker="o", label="MO Q-learning", zorder=5)
        if len(pf_mo) > 1:
            axes[0].plot(x_mo, y_mo, linewidth=2, alpha=0.9)

    if pf_owa:
        x_owa, y_owa = _split_xy(pf_owa)
        axes[0].scatter(x_owa, y_owa, s=100, marker="s", label="OWA Q-learning", zorder=6)
        if len(pf_owa) > 1:
            axes[0].plot(x_owa, y_owa, linewidth=2, alpha=0.9)

    if pf_pql:
        x_pql, y_pql = _split_xy(pf_pql)
        axes[0].scatter(x_pql, y_pql, s=100, marker="D", label="Pareto Q-learning", zorder=7)
        if len(pf_pql) > 1:
            axes[0].plot(x_pql, y_pql, linewidth=2, alpha=0.9)

    if pf_cheb:
        x_cheb, y_cheb = _split_xy(pf_cheb)
        axes[0].scatter(
            x_cheb,
            y_cheb,
            s=140,
            marker="x",
            linewidths=2.5,
            label="Chebyshev Q-learning",
            zorder=20
        )
        if len(pf_cheb) > 1:
            axes[0].plot(x_cheb, y_cheb, linewidth=2, alpha=0.9)

    axes[0].set_xlabel("- Time Cost")
    axes[0].set_ylabel("Treasure Value")
    axes[0].set_title("Pareto Front of Final Learned Policies")
    axes[0].grid(True)
    axes[0].legend()

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
    plt.savefig("results/hypervolume_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()