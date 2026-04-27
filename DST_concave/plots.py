"""
plots.py
========
Plotting module for all MORL experiments.

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


# =============================================================================
# Low-level helpers
# =============================================================================

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


def _save(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → saved {path}")


# =============================================================================
# Smart text-box placement
# Divides the axes into 4 quadrants, counts how many data points fall in each,
# and places the box in the quadrant with the fewest points.
# Works for HV, IGD and Epsilon right panels.
# =============================================================================

def _smart_text_box(ax, text, xs, ys, fontsize=11):
    """
    Place a labelled text box in the corner of `ax` that contains the fewest
    data points.  xs, ys are the data coordinates already on the axes.

    Quadrant map (axes-fraction space):
        TL (0,0.5)-(0.5,1)   TR (0.5,0.5)-(1,1)
        BL (0,0)  -(0.5,0.5) BR (0.5,0) -(1,0.5)
    """
    if len(xs) == 0:
        # No data – default to bottom-right
        _corner_text(ax, text, "BR", fontsize)
        return

    # Convert data coords to axes-fraction using current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0] if xlim[1] != xlim[0] else 1.0
    yspan = ylim[1] - ylim[0] if ylim[1] != ylim[0] else 1.0

    fx = [(x - xlim[0]) / xspan for x in xs]
    fy = [(y - ylim[0]) / yspan for y in ys]

    counts = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
    for x, y in zip(fx, fy):
        if   x < 0.5 and y >= 0.5: counts["TL"] += 1
        elif x >= 0.5 and y >= 0.5: counts["TR"] += 1
        elif x < 0.5 and y < 0.5:  counts["BL"] += 1
        else:                        counts["BR"] += 1

    best = min(counts, key=counts.get)
    _corner_text(ax, text, best, fontsize)


def _corner_text(ax, text, corner, fontsize=11):
    """Place text in the given corner ('TL','TR','BL','BR') of the axes."""
    pad = 0.03
    positions = {
        "BR": (1 - pad, pad,       "right", "bottom"),
        "BL": (pad,     pad,       "left",  "bottom"),
        "TR": (1 - pad, 1 - pad,   "right", "top"),
        "TL": (pad,     1 - pad,   "left",  "top"),
    }
    x, y, ha, va = positions[corner]
    ax.text(x, y, text,
            transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa",
                      alpha=0.95),
            zorder=20)


# =============================================================================
# HV staircase
# =============================================================================

def _hv_staircase(pts_max, ref_point=_REF_HV):
    """
    Return (xs, ys) of the closed staircase polygon for the dominated region.

    pts_max   – list of (x, y) in maximisation form (x = −time_cost, y = treasure)
    ref_point – (rx, ry): worst-case reference (rx is the leftmost boundary)

    The polygon traces:
        start at top-left of first point → step right & down across all points
        → close along the bottom back to start.
    """
    if not pts_max:
        return [], []

    # Sort ascending by x (left to right); Pareto front has y descending
    pts = sorted(pts_max, key=lambda p: p[0])
    rx, ry = ref_point

    xs, ys = [], []

    # Start: from reference x up to first point's y
    xs.append(rx);       ys.append(pts[0][1])   # top-left corner
    xs.append(pts[0][0]); ys.append(pts[0][1])  # horizontal to first point

    for i in range(len(pts)):
        xi, yi = pts[i]
        # Drop vertically to the next point's y (or ry for last point)
        y_next = pts[i + 1][1] if i + 1 < len(pts) else ry
        xs.append(xi); ys.append(y_next)

        if i + 1 < len(pts):
            # Move horizontally right to next point's x
            xs.append(pts[i + 1][0]); ys.append(y_next)

    # Close along the bottom back to the reference x
    xs.append(rx); ys.append(ry)
    xs.append(rx); ys.append(pts[0][1])  # close polygon back to start

    return xs, ys


# =============================================================================
# 1.  PARETO FRONT 
# =============================================================================

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


# =============================================================================
# 2.  HYPERVOLUME
# =============================================================================

def _plot_hv(all_pts_max, hv_ts_dict, hv_pts_dict, key_list,
             algo_key, title_prefix, save_path):
    """
    Left  – HV over timesteps.
    Right – Correct rectangular staircase (like Image 3), with smart-placed
            HV value box that never overlaps data or legend.
    """
    style  = _ALGO_STYLE[algo_key]
    colour = style["color"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

    # Left: HV over time
    for key in key_list:
        if key in hv_ts_dict and key in hv_pts_dict:
            ax1.plot(hv_ts_dict[key], hv_pts_dict[key],
                     lw=1.4, alpha=0.75, label=str(key))
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

    all_x, all_y = [], []
    if pf:
        xs, ys = _hv_staircase(pf, _REF_HV)
        ax2.fill(xs, ys, color=colour, alpha=0.20,
                 label="Dominated area (HV)", zorder=1)
        ax2.plot(xs, ys, color=colour, lw=1.2, ls="--", zorder=2)
        ax2.scatter(*_xy(pf), color=colour, s=70, zorder=5,
                    label="Learned front")
        all_x, all_y = _xy(pf)

    ax2.scatter(*_REF_HV, color="red", marker="x", s=100, lw=2.5,
                label=f"HV ref point {_REF_HV}", zorder=6)
    all_x = list(all_x) + [_REF_HV[0]]
    all_y = list(all_y) + [_REF_HV[1]]

    hv_final = compute_hypervolume_2d(pf, _REF_HV) if pf else 0.0

    ax2.set_xlabel("− Time Cost",    fontsize=12)
    ax2.set_ylabel("Treasure Value", fontsize=12)
    ax2.set_title(f"{title_prefix}\nHV — Dominated Area in Objective Space",
                  fontsize=12, pad=8)
    # Legend fixed to upper-right so text box can avoid it
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, **_GRID_KW)

    # Draw first so limits are set, then place smart text box
    plt.tight_layout()
    _smart_text_box(ax2, f"HV = {hv_final:.4f}", all_x, all_y)

    _save(fig, save_path)


# =============================================================================
# 3.  IGD
# =============================================================================

def _plot_igd(all_pts_max, igd_ts_dict, igd_pts_dict, key_list,
              algo_key, title_prefix, save_path):
    """
    Left  – IGD over timesteps.
    Right – Arrows from every true-front point to nearest learned point,
            plus smart-placed IGD value box (like Image 2).
    """
    style  = _ALGO_STYLE[algo_key]
    colour = style["color"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

    #Left: IGD over time
    for key in key_list:
        if key in igd_ts_dict and key in igd_pts_dict:
            ax1.plot(igd_ts_dict[key], igd_pts_dict[key],
                     lw=1.4, alpha=0.75, label=str(key))
    ax1.set_xlabel("Timestep", fontsize=12)
    ax1.set_ylabel("IGD",      fontsize=12)
    ax1.set_title(f"{title_prefix}\nIGD vs Timestep  (lower = better)",
                  fontsize=12, pad=8)
    ax1.legend(fontsize=7, title="Weights", ncol=2)
    ax1.grid(True, **_GRID_KW)

    # ── Right: arrow visualisation ────────────────────────────────────────────
    tf     = get_true_reference_pf()
    pf     = extract_pareto_front(all_pts_max) if all_pts_max else []
    pf_arr = np.array(pf) if pf else None

    if tf:
        ax2.plot(*_xy(tf), color="#555555", lw=2.0, ls="--",
                 zorder=2, label="True front")

    all_x, all_y = [], []
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
        all_x, all_y = _xy(pf)

    # IGD value
    from utils import compute_igd
    igd_val = compute_igd(tf, pf) if (tf and pf) else 0.0

    ax2.set_xlabel("− Time Cost",    fontsize=12)
    ax2.set_ylabel("Treasure Value", fontsize=12)
    ax2.set_title(f"{title_prefix}\nIGD — True front → Nearest Learned Point",
                  fontsize=12, pad=8)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, **_GRID_KW)

    plt.tight_layout()
    _smart_text_box(ax2, f"IGD = {igd_val:.4f}", list(all_x), list(all_y))

    _save(fig, save_path)


# =============================================================================
# 4.  EPSILON INDICATOR
# =============================================================================

def _plot_epsilon(all_pts_max, eps_ts_dict, eps_pts_dict, key_list,
                  algo_key, title_prefix, save_path):
    """
    Left  – Epsilon over timesteps.
    Right – Learned front (solid) + epsilon-shifted version (faded markers)
            connected by thin lines (like Image 6), with smart-placed ε box.
    """
    style  = _ALGO_STYLE[algo_key]
    colour = style["color"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

    # ── Left: epsilon over time ───────────────────────────────────────────────
    for key in key_list:
        if key in eps_ts_dict and key in eps_pts_dict:
            ax1.plot(eps_ts_dict[key], eps_pts_dict[key],
                     lw=1.4, alpha=0.75, label=str(key))
    ax1.set_xlabel("Timestep",           fontsize=12)
    ax1.set_ylabel("Epsilon indicator",  fontsize=12)
    ax1.set_title(f"{title_prefix}\nEpsilon vs Timestep  (lower = better)",
                  fontsize=12, pad=8)
    ax1.legend(fontsize=7, title="Weights", ncol=2)
    ax1.grid(True, **_GRID_KW)

    # ── Right: shift visualisation ────────────────────────────────────────────
    tf     = get_true_reference_pf()
    pf     = extract_pareto_front(all_pts_max) if all_pts_max else []
    pf_arr = np.array(pf) if pf else None

    if tf:
        ax2.plot(*_xy(tf), color="#555555", lw=2.0, ls="--",
                 zorder=2, label="True front")

    eps_val = 0.0
    all_x, all_y = [], []

    if pf_arr is not None and len(pf_arr) > 0:
        from utils import compute_epsilon_indicator
        eps_val = compute_epsilon_indicator(tf, pf)

        # Original learned front
        ax2.scatter(*_xy(pf), color=colour, s=70, zorder=5,
                    label="Learned front")

        # Epsilon-shifted front (faded)
        pf_shifted = [(x + eps_val, y + eps_val) for x, y in pf]
        ax2.scatter(*_xy(pf_shifted), color=colour, s=40, alpha=0.4,
                    marker="o", zorder=4,
                    label=f"Front + ε ({eps_val:.3f})")

        # Connect original → shifted with thin lines
        for (x0, y0), (x1, y1) in zip(pf, pf_shifted):
            ax2.plot([x0, x1], [y0, y1],
                     color=colour, lw=0.8, alpha=0.5, zorder=3)

        # Collect all plotted x/y for smart placement
        sx, sy = _xy(pf_shifted)
        all_x = list(_xy(pf)[0]) + list(sx)
        all_y = list(_xy(pf)[1]) + list(sy)

    ax2.set_xlabel("− Time Cost",    fontsize=12)
    ax2.set_ylabel("Treasure Value", fontsize=12)
    ax2.set_title(f"{title_prefix}\nEpsilon — Required Shift to Cover True Front",
                  fontsize=12, pad=8)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, **_GRID_KW)

    plt.tight_layout()
    _smart_text_box(ax2, f"ε = {eps_val:.4f}", all_x, all_y)

    _save(fig, save_path)


# =============================================================================
# Public per-algorithm entry points  (all 4 algorithms share _plot_all_for_algo)
# =============================================================================

def _plot_all_for_algo(algo_key, title_prefix, prefix,
                       all_pts_max,
                       hv_ts, hv_pts,
                       igd_ts, igd_pts,
                       eps_ts, eps_pts,
                       key_list):
    """Saves pareto_front, hv, igd, and epsilon plots for one algorithm."""
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


# =============================================================================
# Comparison plots
# =============================================================================

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

    # ── A. HV over time ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(hv_ts_mo[weights_list[0]],    _avg(hv_pts_mo,   weights_list),
            lw=2, color=_lc("mo"),   label=_lb("mo"))
    ax.plot(hv_ts_owa[owa_settings[0]],   _avg(hv_pts_owa,  owa_settings),
            lw=2, color=_lc("owa"),  label=_lb("owa"))
    ax.plot(hv_ts_cheb[cheb_settings[0]], _avg(hv_pts_cheb, cheb_settings),
            lw=2, color=_lc("cheb"), label=_lb("cheb"))
    ax.plot(hv_ts_pql, hv_pts_pql,
            lw=2, color=_lc("pql"),  label=_lb("pql"))
    ax.set_xlabel("Timestep",    fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title("HV Comparison — All Algorithms\n"
                 "(averaged over weight settings; higher = better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, "results/comparison_hv.png")

    # ── B. HV objective-space (staircases) ────────────────────────────────────
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

    # ── C. IGD over time ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(igd_ts_mo[weights_list[0]],    _avg(igd_pts_mo,   weights_list),
            lw=2, color=_lc("mo"),   label=_lb("mo"))
    ax.plot(igd_ts_owa[owa_settings[0]],   _avg(igd_pts_owa,  owa_settings),
            lw=2, color=_lc("owa"),  label=_lb("owa"))
    ax.plot(igd_ts_cheb[cheb_settings[0]], _avg(igd_pts_cheb, cheb_settings),
            lw=2, color=_lc("cheb"), label=_lb("cheb"))
    ax.plot(igd_ts_pql, igd_pts_pql,
            lw=2, color=_lc("pql"),  label=_lb("pql"))
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("IGD",      fontsize=12)
    ax.set_title("IGD Comparison — All Algorithms\n"
                 "(averaged over weight settings; lower = better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, "results/comparison_igd.png")

    # ── D. Epsilon over time ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(eps_ts_mo[weights_list[0]],    _avg(eps_pts_mo,   weights_list),
            lw=2, color=_lc("mo"),   label=_lb("mo"))
    ax.plot(eps_ts_owa[owa_settings[0]],   _avg(eps_pts_owa,  owa_settings),
            lw=2, color=_lc("owa"),  label=_lb("owa"))
    ax.plot(eps_ts_cheb[cheb_settings[0]], _avg(eps_pts_cheb, cheb_settings),
            lw=2, color=_lc("cheb"), label=_lb("cheb"))
    ax.plot(eps_ts_pql, eps_pts_pql,
            lw=2, color=_lc("pql"),  label=_lb("pql"))
    ax.set_xlabel("Timestep",          fontsize=12)
    ax.set_ylabel("Epsilon indicator", fontsize=12)
    ax.set_title("Epsilon Comparison — All Algorithms\n"
                 "(averaged over weight settings; lower = better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _save(fig, "results/comparison_epsilon.png")