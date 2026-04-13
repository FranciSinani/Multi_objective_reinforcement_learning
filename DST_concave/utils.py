import numpy as np


def extract_pareto_front(points):
    """
    points must be in maximization form:
    (x, y) where bigger is better for both objectives
    """
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if q == p:
                continue
            if (q[0] >= p[0] and q[1] >= p[1]) and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    unique = []
    for p in pareto:
        if p not in unique:
            unique.append(p)

    return sorted(unique, key=lambda x: x[0])


def compute_hypervolume_2d(points, ref_point=(-100, 0)):
    """
    2D hypervolume for maximization.
    points: list of (x, y)
    ref_point: dominated reference point
    """
    if not points:
        return 0.0

    pareto = extract_pareto_front(points)
    pareto = sorted(pareto, key=lambda p: p[0])

    hv = 0.0
    prev_x = ref_point[0]

    for x, y in pareto:
        width = x - prev_x
        height = y - ref_point[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x

    return hv


def dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def get_non_dominated(vectors):
    nd = []
    for v in vectors:
        dominated_flag = False
        for u in vectors:
            if u == v:
                continue
            if dominates(u, v):
                dominated_flag = True
                break
        if not dominated_flag and v not in nd:
            nd.append(v)
    return nd


def linear_utility(vec, weights):
    """
    vec and weights must both be length-2 tuples
    vec must be in maximization form: (obj1, obj2)
    """
    return weights[0] * vec[0] + weights[1] * vec[1]


def expected_utility_metric(point_set, weight_list):
    """
    point_set: list of points in maximization form
               example: [(-time_cost, treasure), ...]
    weight_list: list of preference weights
                 example: [(0.9, 0.1), (0.8, 0.2), ...]

    Computes:
        average_w [ max_p u(p, w) ]
    """
    if not point_set:
        return 0.0

    best_utilities = []
    for w in weight_list:
        best_u = max(linear_utility(p, w) for p in point_set)
        best_utilities.append(best_u)

    return float(np.mean(best_utilities))


def dict_points_to_maximize(points_dict):
    """
    Converts dict values like:
        weights -> (time_cost, treasure)
    or
        weights -> [(time_cost, treasure), ...]
    into:
        [(-time_cost, treasure), ...]
    """
    points = []

    for value in points_dict.values():
        if value is None:
            continue

        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], (int, float, np.integer, np.floating))
            and isinstance(value[1], (int, float, np.integer, np.floating))
        ):
            time_cost, treasure = value
            points.append((-time_cost, treasure))

        elif isinstance(value, (list, tuple)):
            for item in value:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[0], (int, float, np.integer, np.floating))
                    and isinstance(item[1], (int, float, np.integer, np.floating))
                ):
                    time_cost, treasure = item
                    points.append((-time_cost, treasure))

    return points


def list_points_to_maximize(points_list):
    """
    Converts:
        [(time_cost, treasure), ...]
    into:
        [(-time_cost, treasure), ...]
    """
    points = []

    for item in points_list:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], (int, float, np.integer, np.floating))
            and isinstance(item[1], (int, float, np.integer, np.floating))
        ):
            time_cost, treasure = item
            points.append((-time_cost, treasure))

    return points