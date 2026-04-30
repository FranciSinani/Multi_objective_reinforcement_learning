"""
Shared metric and helper functions for all MORL algorithms.

Last modified: 2026-04-28

Coordinate convention throughout this file:
    ALL inputs must be in MAXIMISATION form: (-time_cost, treasure)

Metrics
    HV  - Hypervolume (higher = better)
    IGD - Inverted Generational Distance (lower = better)
    EPS - Additive Epsilon Indicator (lower = better)
    EUM - Expected Utility Metric (higher = better)
"""

import numpy as np

# Dominance & Pareto helpers

def dominates(a, b):
    """True if a dominates b (maximisation: larger is better on all objectives)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def get_non_dominated(vectors):
    """
    Return the non-dominated subset of vectors (maximisation).
    Duplicates are preserved; use extract_pareto_front for deduplication.
    """
    nd = []
    for v in vectors:
        if not any(dominates(u, v) for u in vectors if u is not v):
            nd.append(v)
    return nd


def extract_pareto_front(points):
    """
    Deduplicate and return the non-dominated subset, sorted by x ascending.
    All points must be in maximisation form.
    """
    if not points:
        return []
    pts = list(set(map(tuple, points)))
    pareto = []
    for p in pts:
        if not any(dominates(q, p) for q in pts if q != p):
            pareto.append(p)
    return sorted(pareto, key=lambda x: x[0])


# Reference-point validation

_REF_HV = (-100, 0)   # HV reference point


def validate_ref_point(true_front_max, ref_point=_REF_HV):
    """
    Assert that ref_point is strictly dominated by every point on the true front.
    Raises ValueError if any true-front point does not strictly dominate ref_point,
    which would produce invalid (zero or negative) HV contributions.
    """
    for pt in true_front_max:
        if not all(pt[i] > ref_point[i] for i in range(len(ref_point))):
            raise ValueError(
                f"Reference point {ref_point} is NOT strictly dominated by "
                f"true-front point {pt}. HV would be invalid."
            )

# Hypervolume  (2-D sweep-line, maximisation, original scale)

def compute_hypervolume_2d(points, ref_point=_REF_HV):
    """
    2-D hypervolume indicator.
    points    - list of (x, y) in maximisation form (-time_cost, treasure)
    ref_point - must be strictly dominated by all valid points
    Returns the hypervolume of the dominated region (higher is better).
    """
    if not points:
        return 0.0
    pareto = extract_pareto_front(points)
    if not pareto:
        return 0.0
    hv, prev_x = 0.0, ref_point[0]
    for x, y in pareto:
        width  = x - prev_x
        height = y - ref_point[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x
    return hv


# IGD  

def compute_igd(true_front_max, approx_front_max):
    """
    Inverted Generational Distance (lower is better).

    For each point in the TRUE front, finds the nearest point in the
    approximate front and averages those Euclidean distances.
    """
    from pymoo.indicators.igd import IGD as _IGD

    if not true_front_max or not approx_front_max:
        return float("inf")

    # pymoo uses minimisation convention: negate to convert from maximisation
    pf = -np.array(true_front_max, dtype=float)
    A  = -np.array(approx_front_max, dtype=float)
    return float(_IGD(pf)(A))

# Epsilon additive quality indicator

def compute_epsilon_indicator(true_front_max, approx_front_max):
    """
    Additive epsilon indicator (lower is better).

    Minimum scalar epsilon such that for every point r in the true front,
    there exists a point a in the approximate front satisfying:
        a_i + epsilon >= r_i   for all objectives i  (maximisation)

    Interpretation:
        epsilon = 0   -> approximate front fully covers the true front
        epsilon > 0   -> approximate front needs shifting by epsilon
        epsilon < 0   -> approximate front already exceeds the true front
    """
    if not true_front_max or not approx_front_max:
        return float("inf")

    tf = np.array(true_front_max, dtype=float)
    ap = np.array(approx_front_max, dtype=float)

    # For each reference point r: eps(r) = min over a of max_i(r_i - a_i)
    # Overall epsilon = worst case over all r
    eps_per_ref = []
    for r in tf:
        eps_needed = np.max(r - ap, axis=1)
        eps_per_ref.append(float(eps_needed.min()))
    return float(np.max(eps_per_ref))

# Expected Utility Metric

def linear_utility(vec, weights):
    """Weighted linear utility of a single point (maximisation form)."""
    return weights[0] * vec[0] + weights[1] * vec[1]


def expected_utility_metric(point_set, weight_list):
    """
    For each weight vector, pick the best point in the set and return
    the mean utility across all weight vectors.

    point_set   - list of (x, y) in maximisation form
    weight_list - list of (w1, w2) weight vectors
    Higher is better.
    """
    if not point_set:
        return 0.0
    return float(np.mean([
        max(linear_utility(p, w) for p in point_set)
        for w in weight_list
    ]))

# Coordinate conversions

def to_max_form(time_cost, treasure):
    """
    Convert training form (time_cost, treasure) to maximisation form.
    time_cost is a positive step count; we negate it so larger = better.
    Result: (-time_cost, treasure) — both axes larger-is-better.
    """
    return (-time_cost, treasure)


def dict_points_to_maximize(points_dict):
    """
    Convert a weight-keyed dict of (time_cost, treasure) points to a flat
    list in maximisation form (-time_cost, treasure).
    Values may be a single (time_cost, treasure) tuple or a list of such tuples.
    """
    points = []
    for value in points_dict.values():
        if value is None:
            continue
        if (isinstance(value, tuple) and len(value) == 2
                and isinstance(value[0], (int, float, np.integer, np.floating))):
            points.append(to_max_form(*value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    points.append(to_max_form(*item))
    return points


def list_points_to_maximize(points_list):
    """
    Convert [(time_cost, treasure), ...] to [(-time_cost, treasure), ...].
    Used for PQL which returns a list rather than a dict.
    """
    return [to_max_form(*p) for p in points_list]