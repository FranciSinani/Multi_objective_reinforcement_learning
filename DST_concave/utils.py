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