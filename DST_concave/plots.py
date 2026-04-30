"""
Plotting module for all MORL experiments.

Last modified: 2026-04-28

Output structure per algorithm (e.g. "mo"):
    results/mo_pareto_front.png   – learned front vs true front
    results/mo_hv.png             – HV over time  +  objective-space staircase
    results/mo_igd.png            – IGD over time  +  arrow visualisation
    results/mo_epsilon.png        – Epsilon over time  +  shift visualisation

Comparison output (python main.py compare):
    results/comparison_hv.png     – HV over time for all 4 algorithms
    results/comparison_hv_obj.png – HV objective-space view (final front, staircase)
    results/comparison_igd.png    – IGD over time for all 4 algorithms
    results/comparison_epsilon.png– Epsilon over time for all 4 algorithms
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import uniform_filter1d

from env   import get_true_reference_pf
from utils import extract_pareto_front, compute_hypervolume_2d

os.makedirs("results", exist_ok=True)

# Shared visual constants
_DPI      = 150
_GRID_KW  = dict(color="#cccccc", lw=0.7, alpha=0.8)
_TRUE_KW  = dict(color="#555555", lw=2.0, ls="--", zorder=2, label="True front")
_REF_HV   = (-100, 0)   # hypervolume reference point (maximisation space)

_ALGO_STYLE = {
    "mo":   dict(color="#1f77b4", marker="o", label="MO Q-Learning"),
    "owa":  dict(color="#2ca02c", marker="s", label="OWA Q-Learning"),
    "cheb": dict(color="#9467bd", marker="^", label="Chebyshev Q-Learning"),
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
    plt.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → saved {path}")


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
        ax.plot(*_xy(tf), **_TRUE_KW)

    if all_pts_max:
        ax.scatter(*_xy(all_pts_max), s=50, alpha=0.35, color=colour,
                   marker="o", label="All discovered points", zorder=3)

    pf = extract_pareto_front(all_pts_max)
    if pf:
        ax.plot(*_xy(pf), color=colour, lw=2, alpha=0.9, zorder=4)
        ax.scatter(*_xy(pf), s=110, marker="x", color=colour, lw=2.5,
                   label="Learned front", zorder=5)

    ax.set_xlabel("− Time Cost",    fontsize=12)
    ax.set_ylabel("Treasure Value", fontsize=12)
    ax.set_title(f"{title_prefix}: Pareto Front", fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
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
        ax2.plot(*_xy(tf), **_TRUE_KW)

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
        ax2.plot(*_xy(tf), color="#555555", lw=2.0, ls="--",
                 zorder=2, label="True front")

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
        ax2.plot(*_xy(tf), color="#555555", lw=2.0, ls="--",
                 zorder=2, label="True front")

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

# Public per-algorithm entry points

def _plot_all_for_algo(algo_key, title_prefix, prefix,
                       all_pts_max,
                       hv_ts, hv_pts,
                       igd_ts, igd_pts,
                       eps_ts, eps_pts,
                       key_list):
    _plot_pareto_front(
        all_pts_max, algo_key, title_prefix,
        save_path=f"results/{prefix}_pareto_front.png",
    )
    _plot_hv(
        all_pts_max, hv_ts, hv_pts, key_list,
        algo_key, title_prefix,
        save_path=f"results/{prefix}_hv.png",
    )
    _plot_igd(
        all_pts_max, igd_ts, igd_pts, key_list,
        algo_key, title_prefix,
        save_path=f"results/{prefix}_igd.png",
    )
    _plot_epsilon(
        all_pts_max, eps_ts, eps_pts, key_list,
        algo_key, title_prefix,
        save_path=f"results/{prefix}_epsilon.png",
    )


def plot_mo_q_results(all_ep, hv_ts, hv_pts, igd_ts, igd_pts,
                      eps_ts, eps_pts, weights_list):
    _plot_all_for_algo(
        "mo", "MO Q-Learning", "mo",
        _dict_to_max(all_ep),
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts,
        weights_list,
    )


def plot_owa_q_results(all_ep, hv_ts, hv_pts, igd_ts, igd_pts,
                       eps_ts, eps_pts, owa_settings):
    _plot_all_for_algo(
        "owa", "OWA Q-Learning", "owa",
        _dict_to_max(all_ep),
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts,
        owa_settings,
    )


def plot_chebyshev_q_results(all_ep, hv_ts, hv_pts, igd_ts, igd_pts,
                              eps_ts, eps_pts, cheb_settings):
    _plot_all_for_algo(
        "cheb", "Chebyshev Q-Learning", "cheb",
        _dict_to_max(all_ep),
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts,
        cheb_settings,
    )


def plot_pql_results(ep_points, hv_ts, hv_pts, igd_ts, igd_pts,
                     eps_ts, eps_pts):
    _plot_all_for_algo(
        "pql", "Pareto Q-Learning", "pql",
        _list_to_max(ep_points),
        {"pql": hv_ts},  {"pql": hv_pts},
        {"pql": igd_ts}, {"pql": igd_pts},
        {"pql": eps_ts}, {"pql": eps_pts},
        ["pql"],
    )

# Comparison plots

def plot_all_comparisons(
    all_ep_mo,   all_ep_owa,   all_ep_cheb,   ep_pql,
    hv_ts_mo,    hv_pts_mo,
    hv_ts_owa,   hv_pts_owa,
    hv_ts_cheb,  hv_pts_cheb,
    hv_ts_pql,   hv_pts_pql,
    igd_ts_mo,   igd_pts_mo,
    igd_ts_owa,  igd_pts_owa,
    igd_ts_cheb, igd_pts_cheb,
    igd_ts_pql,  igd_pts_pql,
    eps_ts_mo,   eps_pts_mo,
    eps_ts_owa,  eps_pts_owa,
    eps_ts_cheb, eps_pts_cheb,
    eps_ts_pql,  eps_pts_pql,
    weights_list, owa_settings, cheb_settings,
):
    pts_mo   = _dict_to_max(all_ep_mo)
    pts_owa  = _dict_to_max(all_ep_owa)
    pts_cheb = _dict_to_max(all_ep_cheb)
    pts_pql  = _list_to_max(ep_pql)

    pf_mo   = extract_pareto_front(pts_mo)
    pf_owa  = extract_pareto_front(pts_owa)
    pf_cheb = extract_pareto_front(pts_cheb)
    pf_pql  = extract_pareto_front(pts_pql)

    def _lc(key): return _ALGO_STYLE[key]["color"]
    def _lb(key): return _ALGO_STYLE[key]["label"]

    #A. HV over time
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(hv_ts_mo[weights_list[0]],
            _smooth(_avg(hv_pts_mo,   weights_list)),
            lw=2, color=_lc("mo"),   label=_lb("mo"))
    ax.plot(hv_ts_owa[owa_settings[0]],
            _smooth(_avg(hv_pts_owa,  owa_settings)),
            lw=2, color=_lc("owa"),  label=_lb("owa"))
    ax.plot(hv_ts_cheb[cheb_settings[0]],
            _smooth(_avg(hv_pts_cheb, cheb_settings)),
            lw=2, color=_lc("cheb"), label=_lb("cheb"))
    ax.plot(hv_ts_pql, _smooth(hv_pts_pql),
            lw=2, color=_lc("pql"),  label=_lb("pql"))
    ax.set_xlabel("Timestep",    fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title("HV Comparison — All Algorithms\n"
                 "(averaged over weight settings; higher = better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, "results/comparison_hv.png")

    #B. HV objective-space (staircases)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    tf = get_true_reference_pf()
    if tf:
        ax.plot(*_xy(tf), **_TRUE_KW)
    for key, pf in [("mo", pf_mo), ("owa", pf_owa),
                    ("cheb", pf_cheb), ("pql", pf_pql)]:
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
    _save(fig, "results/comparison_hv_obj.png")

    # C. IGD over time
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(igd_ts_mo[weights_list[0]],
            _smooth(_avg(igd_pts_mo,   weights_list)),
            lw=2, color=_lc("mo"),   label=_lb("mo"))
    ax.plot(igd_ts_owa[owa_settings[0]],
            _smooth(_avg(igd_pts_owa,  owa_settings)),
            lw=2, color=_lc("owa"),  label=_lb("owa"))
    ax.plot(igd_ts_cheb[cheb_settings[0]],
            _smooth(_avg(igd_pts_cheb, cheb_settings)),
            lw=2, color=_lc("cheb"), label=_lb("cheb"))
    ax.plot(igd_ts_pql, _smooth(igd_pts_pql),
            lw=2, color=_lc("pql"),  label=_lb("pql"))
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("IGD",      fontsize=12)
    ax.set_title("IGD Comparison — All Algorithms\n"
                 "(averaged over weight settings; lower = better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, "results/comparison_igd.png")

    # D. Epsilon over time
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(eps_ts_mo[weights_list[0]],
            _smooth(_avg(eps_pts_mo,   weights_list)),
            lw=2, color=_lc("mo"),   label=_lb("mo"))
    ax.plot(eps_ts_owa[owa_settings[0]],
            _smooth(_avg(eps_pts_owa,  owa_settings)),
            lw=2, color=_lc("owa"),  label=_lb("owa"))
    ax.plot(eps_ts_cheb[cheb_settings[0]],
            _smooth(_avg(eps_pts_cheb, cheb_settings)),
            lw=2, color=_lc("cheb"), label=_lb("cheb"))
    ax.plot(eps_ts_pql, _smooth(eps_pts_pql),
            lw=2, color=_lc("pql"),  label=_lb("pql"))
    ax.set_xlabel("Timestep",          fontsize=12)
    ax.set_ylabel("Epsilon indicator", fontsize=12)
    ax.set_title("Epsilon Comparison — All Algorithms\n"
                 "(averaged over weight settings; lower = better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, "results/comparison_epsilon.png")