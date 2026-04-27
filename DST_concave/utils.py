"""
utils.py
========
Shared metric and helper functions for all MORL algorithms.

Coordinate convention throughout this file:
    ALL inputs must be in MAXIMISATION form: (-time_cost, treasure)
    i.e. both axes are larger-is-better.

Metrics
-------
    HV  - Hypervolume (higher = better)
    IGD - Inverted Generational Distance, normalised (lower = better)
    EPS - Additive Epsilon Indicator, normalised (lower = better)
    EUM - Expected Utility Metric (higher = better)

Normalisation
-------------
    IGD and Epsilon are computed on objectives normalised to [0, 1]
    using the range of the TRUE Pareto front as bounds.
    This prevents the large x-axis range (-100 to -1) from dominating
    Euclidean distance calculations and makes metrics scale-independent.
    HV is NOT normalised so the absolute dominated area remains interpretable.
"""

import numpy as np


# =============================================================================
# Dominance & Pareto helpers
# =============================================================================

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


# =============================================================================
# Reference-point validation
# =============================================================================

_REF_HV = (-100, 0)   # HV reference point in maximisation space


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


# =============================================================================
# Hypervolume  (2-D sweep-line, maximisation, original scale)
# =============================================================================

def compute_hypervolume_2d(points, ref_point=_REF_HV):
    """
    2-D hypervolume indicator (maximisation, sweep-line).

    Computed in the ORIGINAL (un-normalised) objective space so that
    the absolute dominated area is interpretable and consistent.

    points    - list of (x, y) in maximisation form (-time_cost, treasure)
    ref_point - must be strictly dominated by all valid points
    Returns a scalar (higher is better).
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


# =============================================================================
# Normalisation helper for IGD and Epsilon
# =============================================================================

def _normalise_to_front(points, true_front_max):
    """
    Normalise points to [0,1]^2 using the range of the true Pareto front.

    Both objectives are scaled independently:
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)

    Using the TRUE front range as bounds ensures:
      - A perfect approximation maps to the same normalised coords as the true front
      - Distance metrics are not dominated by objectives with large absolute ranges
      - Results are comparable across algorithms and environments

    Returns a list of (x_norm, y_norm) tuples.
    """
    if not points or not true_front_max:
        return []

    tf = np.array(true_front_max, dtype=float)
    x_min, x_max = tf[:, 0].min(), tf[:, 0].max()
    y_min, y_max = tf[:, 1].min(), tf[:, 1].max()

    if x_max - x_min < 1e-12 or y_max - y_min < 1e-12:
        raise ValueError(
            "True Pareto front has zero range on at least one objective. "
            "Normalisation is undefined for a degenerate front."
        )

    arr = np.array(points, dtype=float)
    arr[:, 0] = (arr[:, 0] - x_min) / (x_max - x_min)
    arr[:, 1] = (arr[:, 1] - y_min) / (y_max - y_min)
    return [tuple(row) for row in arr]


# =============================================================================
# IGD  (normalised, pymoo)
# =============================================================================

def compute_igd(true_front_max, approx_front_max):
    """
    Inverted Generational Distance (lower is better).

    For each point in the TRUE front, finds the nearest point in the
    approximate front and averages those Euclidean distances.

    Both inputs must be in maximisation form: (-time_cost, treasure).

    NORMALISATION: Both fronts are normalised to [0,1]^2 using the true
    front's range before computing distances. This prevents the x-axis
    range (-100 to -1) from dominating distance calculations and makes
    IGD a dimensionless, scale-independent metric.

    Returns float('inf') if either front is empty.
    """
    from pymoo.indicators.igd import IGD as _IGD

    if not true_front_max or not approx_front_max:
        return float("inf")

    tf_norm = _normalise_to_front(true_front_max,   true_front_max)
    ap_norm = _normalise_to_front(approx_front_max, true_front_max)

    # pymoo uses minimisation convention: negate to convert from maximisation
    pf = -np.array(tf_norm, dtype=float)
    A  = -np.array(ap_norm,  dtype=float)
    return float(_IGD(pf)(A))


# =============================================================================
# Epsilon additive quality indicator  (normalised)
# =============================================================================

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

    Both inputs must be in maximisation form: (-time_cost, treasure).

    NORMALISATION: Both fronts are normalised to [0,1]^2 using the true
    front's range. This makes epsilon a dimensionless fraction of the
    front's extent, directly comparable across objectives and algorithms.

    Returns float('inf') if either front is empty.
    """
    if not true_front_max or not approx_front_max:
        return float("inf")

    tf_norm = np.array(_normalise_to_front(true_front_max,   true_front_max), dtype=float)
    ap_norm = np.array(_normalise_to_front(approx_front_max, true_front_max), dtype=float)

    # For each reference point r: eps(r) = min over a of max_i(r_i - a_i)
    # Overall epsilon = worst case over all r
    eps_per_ref = []
    for r in tf_norm:
        eps_needed = np.max(r - ap_norm, axis=1)
        eps_per_ref.append(float(eps_needed.min()))
    return float(np.max(eps_per_ref))


# =============================================================================
# Expected Utility Metric
# =============================================================================

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


# =============================================================================
# Coordinate conversions
# =============================================================================

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