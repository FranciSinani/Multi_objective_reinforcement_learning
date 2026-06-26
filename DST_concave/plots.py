"""
Plotting module for all MORL experiments.

Last modified: 2026-04-28

Each method writes to its own subfolder under results/. Comparison plots are
written to results/comparisons/.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import uniform_filter1d

from env   import get_true_reference_pf
from utils import (
    compute_epsilon_indicator,
    compute_hypervolume_2d,
    compute_igd,
    extract_pareto_front,
)

os.makedirs("results", exist_ok=True)

# Shared visual constants
_DPI      = 150
_GRID_KW  = dict(color="#cccccc", lw=0.7, alpha=0.8)
_TRUE_KW = dict(
    color="#555555",
    s=65,
    marker="^",
    facecolors="none",
    linewidths=1.6,
    zorder=2,
    label="True Pareto solutions",
)
_REF_HV   = (-100, 0)   # hypervolume reference point (maximisation space)

_ALGO_STYLE = {
    "mo":   dict(color="#1f77b4", marker="o", label="MO Q-Learning"),
    "owa":  dict(color="#2ca02c", marker="s", label="OWA Q-Learning"),
    "cheb": dict(color="#9467bd", marker="^", label="Chebyshev Q-Learning"),
    "choquet": dict(color="#ff7f0e", marker="P", label="Choquet Integral Q-Learning"),
    "pql":  dict(color="#d62728", marker="D", label="Pareto Q-Learning"),
}

# Smoothing window for timestep plots 
_SMOOTH_WINDOW = 21

# Low-level helpers

def _xy(pts):
    return [p[0] for p in pts], [p[1] for p in pts]


def _to_max(time_cost, treasure):
    return (-time_cost, treasure)


def _dict_to_max(d):
    """Dict {weights: (tc, tr) or [(tc,tr),...]}  →  [(-tc, tr), ...]"""
    pts = []
    for v in d.values():
        if v is None:
            continue
        if (isinstance(v, tuple) and len(v) == 2
                and isinstance(v[0], (int, float, np.integer, np.floating))):
            pts.append(_to_max(*v))
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    pts.append(_to_max(*item))
    return pts


def _list_to_max(lst):
    return [_to_max(*p) for p in lst]


def _avg(d, keys):
    """Average metric curves across weight settings."""
    return np.mean(np.array([d[k] for k in keys]), axis=0)


def _smooth(y, window=_SMOOTH_WINDOW):
    """Apply uniform (box) smoothing to a 1-D array. Safe for short arrays."""
    arr = np.asarray(y, dtype=float)
    if len(arr) < window:
        return arr
    return uniform_filter1d(arr, size=window, mode="nearest")


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def _below_axes_text(ax, text, fontsize=11):
    """
    Place a labelled text box just below the axes (outside the plot area),
    horizontally centred.  This guarantees zero overlap with legend or data.
    """
    ax.text(
        0.5, -0.13, text,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaaaaa", alpha=0.95),
        zorder=20,
    )

# HV staircase

def _hv_staircase(pts_max, ref_point=_REF_HV):
    """
    Return (xs, ys) of the closed staircase polygon for the dominated region.
    """
    if not pts_max:
        return [], []

    pts = sorted(pts_max, key=lambda p: p[0])
    rx, ry = ref_point

    xs, ys = [], []
    xs.append(rx);        ys.append(pts[0][1])
    xs.append(pts[0][0]); ys.append(pts[0][1])

    for i in range(len(pts)):
        xi, yi = pts[i]
        y_next = pts[i + 1][1] if i + 1 < len(pts) else ry
        xs.append(xi); ys.append(y_next)
        if i + 1 < len(pts):
            xs.append(pts[i + 1][0]); ys.append(y_next)

    xs.append(rx); ys.append(ry)
    xs.append(rx); ys.append(pts[0][1])
    return xs, ys

# 1.  PARETO FRONT


def _plot_pareto_front(all_pts_max, algo_key, title_prefix, save_path):
    style  = _ALGO_STYLE[algo_key]
    colour = style["color"]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    tf = get_true_reference_pf()
    if tf:
        ax.scatter(*_xy(tf), **_TRUE_KW)

    if all_pts_max:
        ax.scatter(*_xy(all_pts_max), s=50, alpha=0.35, color=colour,
                   marker="o", label="All discovered points", zorder=3)

    pf = extract_pareto_front(all_pts_max)
    if pf:
        ax.scatter(*_xy(pf), s=110, marker="x", color=colour, lw=2.5,
                   label="Learned solutions", zorder=5)

    ax.set_xlabel("− Time Cost",    fontsize=12)
    ax.set_ylabel("Treasure Value", fontsize=12)
    ax.set_title(title_prefix, fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, save_path)


def _coverage(true_front, learned_front, tolerance=0.5):
    covered = [
        true_point
        for true_point in true_front
        if any(
            np.linalg.norm(
                np.asarray(true_point) - np.asarray(learned_point)
            ) <= tolerance
            for learned_point in learned_front
        )
    ]
    return len(covered), len(true_front)


def _plot_archive_and_final_front(
    archive_pts_max,
    final_pts_max,
    algo_key,
    title_prefix,
    save_path,
):
    colour = _ALGO_STYLE[algo_key]["color"]
    true_front = get_true_reference_pf()
    archive_front = extract_pareto_front(archive_pts_max)
    final_front = extract_pareto_front(final_pts_max)
    archive_coverage = _coverage(true_front, archive_front)
    final_coverage = _coverage(true_front, final_front)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    if true_front:
        ax.scatter(*_xy(true_front), **_TRUE_KW)

    if archive_front:
        ax.scatter(
            *_xy(archive_front),
            color=colour,
            s=70,
            alpha=0.85,
            label=(
                "Solutions found during training "
                f"({archive_coverage[0]}/{archive_coverage[1]} covered)"
            ),
            zorder=3,
        )

    if final_pts_max:
        ax.scatter(
            *_xy(final_pts_max),
            color="#d62728",
            marker="x",
            s=110,
            linewidth=2.5,
            label=(
                "Final policies solutions "
                f"({final_coverage[0]}/{final_coverage[1]} covered)"
            ),
            zorder=4,
        )

    ax.set_xlabel("- Time Cost", fontsize=12)
    ax.set_ylabel("Treasure Value", fontsize=12)
    ax.set_title(title_prefix, fontsize=13, pad=10)
    ax.grid(True, **_GRID_KW)
    ax.legend(fontsize=9)
    _save(fig, save_path)


# 2.  HYPERVOLUME

def _plot_hv(all_pts_max, hv_ts_dict, hv_pts_dict, key_list,
             algo_key, title_prefix, save_path):
    style  = _ALGO_STYLE[algo_key]
    colour = style["color"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

    # Left: smoothed HV over time
    for key in key_list:
        if key in hv_ts_dict and key in hv_pts_dict:
            ys = _smooth(hv_pts_dict[key])
            ax1.plot(hv_ts_dict[key], ys, lw=1.4, alpha=0.75, label=str(key))
    ax1.set_xlabel("Timestep",    fontsize=12)
    ax1.set_ylabel("Hypervolume", fontsize=12)
    ax1.set_title(f"{title_prefix}\nHV vs Timestep  (higher = better)",
                  fontsize=12, pad=8)
    ax1.legend(fontsize=7, title="Weights", ncol=2)
    ax1.grid(True, **_GRID_KW)

    # Right: staircase
    tf = get_true_reference_pf()
    if tf:
        ax2.scatter(*_xy(tf), **_TRUE_KW)

    pf = extract_pareto_front(all_pts_max) if all_pts_max else []
    if pf:
        xs, ys = _hv_staircase(pf, _REF_HV)
        ax2.fill(xs, ys, color=colour, alpha=0.20,
                 label="Dominated area (HV)", zorder=1)
        ax2.plot(xs, ys, color=colour, lw=1.2, ls="--", zorder=2)
        ax2.scatter(*_xy(pf), color=colour, s=70, zorder=5,
                    label="Learned front")

    ax2.scatter(*_REF_HV, color="red", marker="x", s=100, lw=2.5,
                label=f"HV ref point {_REF_HV}", zorder=6)

    hv_final = compute_hypervolume_2d(pf, _REF_HV) if pf else 0.0

    ax2.set_xlabel("− Time Cost",    fontsize=12)
    ax2.set_ylabel("Treasure Value", fontsize=12)
    ax2.set_title(f"{title_prefix}\nHV — Dominated Area in Objective Space",
                  fontsize=12, pad=8)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, **_GRID_KW)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _below_axes_text(ax2, f"HV = {hv_final:.4f}")

    _save(fig, save_path)


# 3.  IGD

def _plot_igd(all_pts_max, igd_ts_dict, igd_pts_dict, key_list,
              algo_key, title_prefix, save_path):
    style  = _ALGO_STYLE[algo_key]
    colour = style["color"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

    # Left: smoothed IGD over time
    for key in key_list:
        if key in igd_ts_dict and key in igd_pts_dict:
            ys = _smooth(igd_pts_dict[key])
            ax1.plot(igd_ts_dict[key], ys, lw=1.4, alpha=0.75, label=str(key))
    ax1.set_xlabel("Timestep", fontsize=12)
    ax1.set_ylabel("IGD",      fontsize=12)
    ax1.set_title(f"{title_prefix}\nIGD vs Timestep  (lower = better)",
                  fontsize=12, pad=8)
    ax1.legend(fontsize=7, title="Weights", ncol=2)
    ax1.grid(True, **_GRID_KW)

    # Right: arrow visualisation
    tf     = get_true_reference_pf()
    pf     = extract_pareto_front(all_pts_max) if all_pts_max else []
    pf_arr = np.array(pf) if pf else None

    if tf:
        ax2.scatter(*_xy(tf), **_TRUE_KW)

    if pf_arr is not None and len(pf_arr) > 0:
        tf_arr = np.array(tf)
        for r in tf_arr:
            dists  = np.linalg.norm(pf_arr - r, axis=1)
            a_near = pf_arr[dists.argmin()]
            if np.linalg.norm(a_near - r) > 1e-6:
                ax2.annotate(
                    "", xy=a_near, xytext=r,
                    arrowprops=dict(arrowstyle="-|>", color="#888888",
                                    lw=0.9, mutation_scale=7,
                                    shrinkA=3, shrinkB=3),
                    zorder=3,
                )
        ax2.scatter(*_xy(pf), color=colour, s=70, zorder=5,
                    label="Learned front")

    from utils import compute_igd
    igd_val = compute_igd(tf, pf) if (tf and pf) else 0.0

    ax2.set_xlabel("− Time Cost",    fontsize=12)
    ax2.set_ylabel("Treasure Value", fontsize=12)
    ax2.set_title(f"{title_prefix}\nIGD — True front → Nearest Learned Point",
                  fontsize=12, pad=8)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, **_GRID_KW)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _below_axes_text(ax2, f"IGD = {igd_val:.4f}")

    _save(fig, save_path)

# 4.  EPSILON INDICATOR

def _plot_epsilon(all_pts_max, eps_ts_dict, eps_pts_dict, key_list,
                  algo_key, title_prefix, save_path):
    style  = _ALGO_STYLE[algo_key]
    colour = style["color"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

    # Left: smoothed Epsilon over time
    for key in key_list:
        if key in eps_ts_dict and key in eps_pts_dict:
            ys = _smooth(eps_pts_dict[key])
            ax1.plot(eps_ts_dict[key], ys, lw=1.4, alpha=0.75, label=str(key))
    ax1.set_xlabel("Timestep",           fontsize=12)
    ax1.set_ylabel("Epsilon indicator",  fontsize=12)
    ax1.set_title(f"{title_prefix}\nEpsilon vs Timestep  (lower = better)",
                  fontsize=12, pad=8)
    ax1.legend(fontsize=7, title="Weights", ncol=2)
    ax1.grid(True, **_GRID_KW)

    # Right: shift visualisation
    tf     = get_true_reference_pf()
    pf     = extract_pareto_front(all_pts_max) if all_pts_max else []
    pf_arr = np.array(pf) if pf else None

    if tf:
        ax2.scatter(*_xy(tf), **_TRUE_KW)

    eps_val = 0.0
    if pf_arr is not None and len(pf_arr) > 0:
        from utils import compute_epsilon_indicator
        eps_val = compute_epsilon_indicator(tf, pf)

        ax2.scatter(*_xy(pf), color=colour, s=70, zorder=5,
                    label="Learned front")

        pf_shifted = [(x + eps_val, y + eps_val) for x, y in pf]
        ax2.scatter(*_xy(pf_shifted), color=colour, s=40, alpha=0.4,
                    marker="o", zorder=4,
                    label=f"Front + ε ({eps_val:.3f})")

        for (x0, y0), (x1, y1) in zip(pf, pf_shifted):
            ax2.plot([x0, x1], [y0, y1],
                     color=colour, lw=0.8, alpha=0.5, zorder=3)

    ax2.set_xlabel("− Time Cost",    fontsize=12)
    ax2.set_ylabel("Treasure Value", fontsize=12)
    ax2.set_title(f"{title_prefix}\nEpsilon — Required Shift to Cover True Front",
                  fontsize=12, pad=8)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, **_GRID_KW)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _below_axes_text(ax2, f"ε = {eps_val:.4f}")

    _save(fig, save_path)

def _metric_legend(ax, algo_key, key_list):
    if not key_list:
        return
    if algo_key == "choquet":
        title = "Capacities"
    elif algo_key == "pql":
        title = None
    else:
        title = "Weights"
    ax.legend(
        fontsize=7,
        title=title,
        ncol=2 if len(key_list) > 5 else 1,
        loc="best",
    )


def _plot_metric_over_time(
    timestep_dict,
    value_dict,
    key_list,
    algo_key,
    ylabel,
    title,
    save_path,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    for key in key_list:
        if key not in timestep_dict or key not in value_dict:
            continue
        ax.plot(
            timestep_dict[key],
            value_dict[key],
            linewidth=1.4,
            alpha=0.85,
            label=str(key),
        )

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(True, **_GRID_KW)
    _metric_legend(ax, algo_key, key_list)
    fig.tight_layout()
    _save(fig, save_path)


def _plot_hv(all_pts_max, hv_ts_dict, hv_pts_dict, key_list,
             algo_key, title_prefix, save_path):
    _plot_metric_over_time(
        hv_ts_dict,
        hv_pts_dict,
        key_list,
        algo_key,
        "Hypervolume",
        "HV vs Timestep (higher = better)",
        save_path,
    )


def _plot_igd(all_pts_max, igd_ts_dict, igd_pts_dict, key_list,
              algo_key, title_prefix, save_path):
    _plot_metric_over_time(
        igd_ts_dict,
        igd_pts_dict,
        key_list,
        algo_key,
        "IGD",
        "IGD vs Timestep (lower = better)",
        save_path,
    )


def _plot_epsilon(all_pts_max, eps_ts_dict, eps_pts_dict, key_list,
                  algo_key, title_prefix, save_path):
    _plot_metric_over_time(
        eps_ts_dict,
        eps_pts_dict,
        key_list,
        algo_key,
        "Epsilon indicator",
        "Epsilon vs Timestep (lower = better)",
        save_path,
    )


# Public per-algorithm entry points

def _plot_all_for_algo(algo_key, title_prefix, prefix,
                       all_pts_max,
                       hv_ts, hv_pts,
                       igd_ts, igd_pts,
                       eps_ts, eps_pts,
                       key_list,
                       output_dir):
    _plot_pareto_front(
        all_pts_max, algo_key, title_prefix,
        save_path=os.path.join(output_dir, f"{prefix}_pareto_front.png"),
    )
    _plot_hv(
        all_pts_max, hv_ts, hv_pts, key_list,
        algo_key, title_prefix,
        save_path=os.path.join(output_dir, f"{prefix}_hv.png"),
    )
    _plot_igd(
        all_pts_max, igd_ts, igd_pts, key_list,
        algo_key, title_prefix,
        save_path=os.path.join(output_dir, f"{prefix}_igd.png"),
    )
    _plot_epsilon(
        all_pts_max, eps_ts, eps_pts, key_list,
        algo_key, title_prefix,
        save_path=os.path.join(output_dir, f"{prefix}_epsilon.png"),
    )


def _value_box(ax, label, value):
    ax.text(
        0.98,
        0.98,
        f"{label} = {value:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec="#888888",
            alpha=0.95,
        ),
        zorder=20,
    )


def save_method_quality_plots(
    method_name,
    final_points_max,
    algo_key,
    output_dir,
    prefix,
):
    """Save final HV, IGD, and epsilon plots for one combined method front."""
    os.makedirs(output_dir, exist_ok=True)
    colour = _ALGO_STYLE[algo_key]["color"]
    true_front = get_true_reference_pf()
    method_front = extract_pareto_front(final_points_max)

    # Hypervolume
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(*_xy(true_front), **_TRUE_KW)
    if method_front:
        xs, ys = _hv_staircase(method_front, _REF_HV)
        ax.fill(xs, ys, color=colour, alpha=0.2, label="Dominated area")
        ax.scatter(
            *_xy(method_front),
            color=colour,
            s=85,
            label="Final policies solutions",
            zorder=4,
        )
    ax.scatter(
        *_REF_HV,
        color="#d62728",
        marker="x",
        s=100,
        linewidth=2.5,
        label=f"Reference point {_REF_HV}",
    )
    hv_value = compute_hypervolume_2d(method_front, _REF_HV)
    _value_box(ax, "Method HV", hv_value)
    ax.set_xlabel("- Time Cost")
    ax.set_ylabel("Treasure Value")
    ax.set_title(f"{method_name} - Hypervolume")
    ax.grid(True, **_GRID_KW)
    ax.legend()
    _save(fig, os.path.join(output_dir, f"{prefix}_hv.png"))

    # IGD
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(*_xy(true_front), **_TRUE_KW)
    if method_front:
        method_array = np.asarray(method_front, dtype=float)
        true_array = np.asarray(true_front, dtype=float)
        lower = true_array.min(axis=0)
        upper = true_array.max(axis=0)
        span = np.where(
            np.isclose(upper - lower, 0.0),
            1.0,
            upper - lower,
        )
        normalized_method = (method_array - lower) / span
        normalized_true = (true_array - lower) / span
        for true_point, normalized_point in zip(
            true_array,
            normalized_true,
        ):
            distances = np.linalg.norm(
                normalized_method - normalized_point,
                axis=1,
            )
            nearest = method_array[distances.argmin()]
            if not np.allclose(true_point, nearest):
                ax.plot(
                    [true_point[0], nearest[0]],
                    [true_point[1], nearest[1]],
                    color="#999999",
                    linewidth=0.9,
                    zorder=1,
                )
        ax.scatter(
            *_xy(method_front),
            color=colour,
            s=85,
            label="Final policies solutions",
            zorder=4,
        )
    igd_value = compute_igd(true_front, method_front)
    _value_box(ax, "Method IGD", igd_value)
    ax.set_xlabel("- Time Cost")
    ax.set_ylabel("Treasure Value")
    ax.set_title(f"{method_name} - IGD")
    ax.grid(True, **_GRID_KW)
    ax.legend()
    _save(fig, os.path.join(output_dir, f"{prefix}_igd.png"))

    # Additive epsilon
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(*_xy(true_front), **_TRUE_KW)
    epsilon_value = compute_epsilon_indicator(true_front, method_front)
    if method_front:
        ax.scatter(
            *_xy(method_front),
            color=colour,
            s=85,
            label="Final policies solutions",
            zorder=4,
        )
        shifted_front = [
            (x + epsilon_value, y + epsilon_value)
            for x, y in method_front
        ]
        ax.scatter(
            *_xy(shifted_front),
            color=colour,
            marker="x",
            s=70,
            alpha=0.65,
            label="Solutions shifted by epsilon",
            zorder=3,
        )
    _value_box(ax, "Method epsilon", epsilon_value)
    ax.set_xlabel("- Time Cost")
    ax.set_ylabel("Treasure Value")
    ax.set_title(f"{method_name} - Epsilon Indicator")
    ax.grid(True, **_GRID_KW)
    ax.legend()
    _save(fig, os.path.join(output_dir, f"{prefix}_epsilon.png"))


def plot_mo_q_results(all_ep, hv_ts, hv_pts, igd_ts, igd_pts,
                      eps_ts, eps_pts, weights_list,
                      final_points=None,
                      output_dir="results/tabular_weighted_sum"):
    archive_pts_max = _dict_to_max(all_ep)
    _plot_all_for_algo(
        "mo", "MO Q-Learning", "mo",
        archive_pts_max,
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts,
        weights_list, output_dir,
    )
    if final_points is not None:
        final_points_max = _dict_to_max(final_points)
        _plot_archive_and_final_front(
            archive_pts_max,
            final_points_max,
            "mo",
            "MO Q-Learning",
            os.path.join(output_dir, "mo_pareto_front.png"),
        )


def plot_owa_q_results(all_ep, hv_ts, hv_pts, igd_ts, igd_pts,
                       eps_ts, eps_pts, owa_settings, final_points=None,
                       output_dir="results/tabular_owa"):
    archive_pts_max = _dict_to_max(all_ep)
    _plot_all_for_algo(
        "owa", "OWA Q-Learning", "owa",
        archive_pts_max,
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts,
        owa_settings, output_dir,
    )
    if final_points is not None:
        final_points_max = _dict_to_max(final_points)
        _plot_archive_and_final_front(
            archive_pts_max,
            final_points_max,
            "owa",
            "OWA Q-Learning",
            os.path.join(output_dir, "owa_pareto_front.png"),
        )


def plot_chebyshev_q_results(all_ep, hv_ts, hv_pts, igd_ts, igd_pts,
                              eps_ts, eps_pts, cheb_settings,
                              final_points=None,
                              output_dir="results/tabular_chebyshev"):
    archive_pts_max = _dict_to_max(all_ep)
    _plot_all_for_algo(
        "cheb", "Chebyshev Q-Learning", "cheb",
        archive_pts_max,
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts,
        cheb_settings, output_dir,
    )
    if final_points is not None:
        final_points_max = _dict_to_max(final_points)
        _plot_archive_and_final_front(
            archive_pts_max,
            final_points_max,
            "cheb",
            "Chebyshev Q-Learning",
            os.path.join(output_dir, "cheb_pareto_front.png"),
        )


def plot_choquet_q_results(all_ep, hv_ts, hv_pts, igd_ts, igd_pts,
                           eps_ts, eps_pts, choquet_settings,
                           final_points=None,
                           output_dir="results/tabular_choquet"):
    archive_pts_max = _dict_to_max(all_ep)
    _plot_all_for_algo(
        "choquet", "Tabular Choquet Q-Learning", "choquet",
        archive_pts_max,
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts,
        choquet_settings, output_dir,
    )
    if final_points is not None:
        final_points_max = _dict_to_max(final_points)
        _plot_archive_and_final_front(
            archive_pts_max,
            final_points_max,
            "choquet",
            "Tabular Choquet Q-Learning",
            os.path.join(output_dir, "choquet_pareto_front.png"),
        )


def plot_pql_results(ep_points, hv_ts, hv_pts, igd_ts, igd_pts,
                     eps_ts, eps_pts,
                     output_dir="results/pareto_q_learning"):
    learned_front = extract_pareto_front(_list_to_max(ep_points))
    _plot_all_for_algo(
        "pql", "Pareto Q-Learning", "pql",
        learned_front,
        {"pql": hv_ts},  {"pql": hv_pts},
        {"pql": igd_ts}, {"pql": igd_pts},
        {"pql": eps_ts}, {"pql": eps_pts},
        ["pql"], output_dir,
    )

    true_front = get_true_reference_pf()
    coverage = _coverage(true_front, learned_front)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    if true_front:
        ax.scatter(
            *_xy(true_front),
            color="#555555",
            marker="^",
            s=150,
            facecolors="none",
            linewidths=1.8,
            label="True Pareto solutions",
            zorder=4,
        )
    if learned_front:
        ax.scatter(
            *_xy(learned_front),
            color=_ALGO_STYLE["pql"]["color"],
            marker="x",
            s=70,
            linewidth=2.3,
            label=(
                "Learned Pareto solutions "
                f"({coverage[0]}/{coverage[1]} covered)"
            ),
            zorder=5,
        )

    ax.set_xlabel("- Time Cost", fontsize=12)
    ax.set_ylabel("Treasure Value", fontsize=12)
    ax.set_title("Pareto Q-Learning", fontsize=13, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, **_GRID_KW)
    _save(fig, os.path.join(output_dir, "pql_pareto_front.png"))

# Comparison plots

def plot_all_comparisons(
    all_ep_mo,   all_ep_owa,   all_ep_cheb,   all_ep_choquet,   ep_pql,
    hv_ts_mo,    hv_pts_mo,
    hv_ts_owa,   hv_pts_owa,
    hv_ts_cheb,  hv_pts_cheb,
    hv_ts_choquet, hv_pts_choquet,
    hv_ts_pql,   hv_pts_pql,
    igd_ts_mo,   igd_pts_mo,
    igd_ts_owa,  igd_pts_owa,
    igd_ts_cheb, igd_pts_cheb,
    igd_ts_choquet, igd_pts_choquet,
    igd_ts_pql,  igd_pts_pql,
    eps_ts_mo,   eps_pts_mo,
    eps_ts_owa,  eps_pts_owa,
    eps_ts_cheb, eps_pts_cheb,
    eps_ts_choquet, eps_pts_choquet,
    eps_ts_pql,  eps_pts_pql,
    weights_list, owa_settings, cheb_settings, choquet_settings,
    output_dir="results/comparisons",
):
    pts_mo   = _dict_to_max(all_ep_mo)
    pts_owa  = _dict_to_max(all_ep_owa)
    pts_cheb = _dict_to_max(all_ep_cheb)
    pts_choquet = _dict_to_max(all_ep_choquet)
    pts_pql  = _list_to_max(ep_pql)

    pf_mo   = extract_pareto_front(pts_mo)
    pf_owa  = extract_pareto_front(pts_owa)
    pf_cheb = extract_pareto_front(pts_cheb)
    pf_choquet = extract_pareto_front(pts_choquet)
    pf_pql  = extract_pareto_front(pts_pql)

    def _lc(key): return _ALGO_STYLE[key]["color"]
    def _lb(key): return _ALGO_STYLE[key]["label"]

    #A. HV over time
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    hv_mo = _smooth(_avg(hv_pts_mo, weights_list))
    hv_owa = _smooth(_avg(hv_pts_owa, owa_settings))
    hv_cheb = _smooth(_avg(hv_pts_cheb, cheb_settings))
    hv_choquet = _smooth(_avg(hv_pts_choquet, choquet_settings))
    ax.plot(hv_ts_mo[weights_list[0]], hv_mo,
            lw=2, color=_lc("mo"),
            label=f"{_lb('mo')} ({hv_mo[-1]:.4f})")
    ax.plot(hv_ts_owa[owa_settings[0]], hv_owa,
            lw=2, color=_lc("owa"),
            label=f"{_lb('owa')} ({hv_owa[-1]:.4f})")
    ax.plot(hv_ts_cheb[cheb_settings[0]], hv_cheb,
            lw=2, color=_lc("cheb"),
            label=f"{_lb('cheb')} ({hv_cheb[-1]:.4f})")
    ax.plot(hv_ts_choquet[choquet_settings[0]], hv_choquet,
            lw=2, color=_lc("choquet"),
            label=f"{_lb('choquet')} ({hv_choquet[-1]:.4f})")
    ax.plot(hv_ts_pql, _smooth(hv_pts_pql),
            lw=2, color=_lc("pql"),
            label=f"{_lb('pql')} ({hv_pts_pql[-1]:.4f})")
    ax.set_xlabel("Timestep",    fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title("HV Comparison — All Algorithms\n"
                 "(averaged over weight settings; higher = better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, os.path.join(output_dir, "comparison_hv.png"))

    #B. HV objective-space (staircases)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    tf = get_true_reference_pf()
    if tf:
        ax.scatter(*_xy(tf), **_TRUE_KW)
    for key, pf in [("mo", pf_mo), ("owa", pf_owa),
                    ("cheb", pf_cheb), ("choquet", pf_choquet),
                    ("pql", pf_pql)]:
        c = _lc(key)
        if pf:
            xs, ys = _hv_staircase(pf, _REF_HV)
            ax.fill(xs, ys, color=c, alpha=0.15, zorder=1)
            ax.plot(xs, ys, color=c, lw=1.0, ls="--", zorder=2)
            ax.scatter(*_xy(pf), color=c, s=60, zorder=5,
                       label=f"{_lb(key)}  HV={compute_hypervolume_2d(pf, _REF_HV):.1f}")
    ax.scatter(*_REF_HV, color="red", marker="x", s=100, lw=2.5,
               label=f"HV ref point {_REF_HV}", zorder=6)
    ax.set_xlabel("− Time Cost",    fontsize=12)
    ax.set_ylabel("Treasure Value", fontsize=12)
    ax.set_title("HV Objective-Space View — Final Fronts\n"
                 "(shaded = dominated area per algorithm)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, **_GRID_KW)
    _save(fig, os.path.join(output_dir, "comparison_hv_obj.png"))

    # C. IGD over time
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    igd_mo = _smooth(_avg(igd_pts_mo, weights_list))
    igd_owa = _smooth(_avg(igd_pts_owa, owa_settings))
    igd_cheb = _smooth(_avg(igd_pts_cheb, cheb_settings))
    igd_choquet = _smooth(_avg(igd_pts_choquet, choquet_settings))
    ax.plot(igd_ts_mo[weights_list[0]], igd_mo,
            lw=2, color=_lc("mo"),
            label=f"{_lb('mo')} ({igd_mo[-1]:.4f})")
    ax.plot(igd_ts_owa[owa_settings[0]], igd_owa,
            lw=2, color=_lc("owa"),
            label=f"{_lb('owa')} ({igd_owa[-1]:.4f})")
    ax.plot(igd_ts_cheb[cheb_settings[0]], igd_cheb,
            lw=2, color=_lc("cheb"),
            label=f"{_lb('cheb')} ({igd_cheb[-1]:.4f})")
    ax.plot(igd_ts_choquet[choquet_settings[0]], igd_choquet,
            lw=2, color=_lc("choquet"),
            label=f"{_lb('choquet')} ({igd_choquet[-1]:.4f})")
    ax.plot(igd_ts_pql, _smooth(igd_pts_pql),
            lw=2, color=_lc("pql"),
            label=f"{_lb('pql')} ({igd_pts_pql[-1]:.4f})")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("IGD",      fontsize=12)
    ax.set_title("IGD Comparison — All Algorithms\n"
                 "(averaged over weight settings; lower = better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, os.path.join(output_dir, "comparison_igd.png"))

    # D. Epsilon over time
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    eps_mo = _smooth(_avg(eps_pts_mo, weights_list))
    eps_owa = _smooth(_avg(eps_pts_owa, owa_settings))
    eps_cheb = _smooth(_avg(eps_pts_cheb, cheb_settings))
    eps_choquet = _smooth(_avg(eps_pts_choquet, choquet_settings))
    ax.plot(eps_ts_mo[weights_list[0]], eps_mo,
            lw=2, color=_lc("mo"),
            label=f"{_lb('mo')} ({eps_mo[-1]:.4f})")
    ax.plot(eps_ts_owa[owa_settings[0]], eps_owa,
            lw=2, color=_lc("owa"),
            label=f"{_lb('owa')} ({eps_owa[-1]:.4f})")
    ax.plot(eps_ts_cheb[cheb_settings[0]], eps_cheb,
            lw=2, color=_lc("cheb"),
            label=f"{_lb('cheb')} ({eps_cheb[-1]:.4f})")
    ax.plot(eps_ts_choquet[choquet_settings[0]], eps_choquet,
            lw=2, color=_lc("choquet"),
            label=f"{_lb('choquet')} ({eps_choquet[-1]:.4f})")
    ax.plot(eps_ts_pql, _smooth(eps_pts_pql),
            lw=2, color=_lc("pql"),
            label=f"{_lb('pql')} ({eps_pts_pql[-1]:.4f})")
    ax.set_xlabel("Timestep",          fontsize=12)
    ax.set_ylabel("Epsilon indicator", fontsize=12)
    ax.set_title("Epsilon Comparison — All Algorithms\n"
                 "(averaged over weight settings; lower = better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, os.path.join(output_dir, "comparison_epsilon.png"))
