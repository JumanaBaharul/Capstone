"""Optimised treatment path generation using graph algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, List, Tuple

import networkx as nx
from shapely.geometry import Polygon


@dataclass
class RouteSegment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    length: float


@dataclass
class TreatmentRoute:
    order: List[str]
    waypoints: List[Tuple[float, float]]
    total_length: float
    segments: List[RouteSegment]


def _build_complete_graph(points: List[Tuple[float, float]]) -> nx.Graph:
    graph = nx.Graph()
    for idx, point in enumerate(points):
        graph.add_node(idx, coord=point)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            length = hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            graph.add_edge(i, j, weight=length)
    return graph


def compute_route(polygons: Iterable[Polygon], start: Tuple[float, float] = (0.0, 0.0)) -> TreatmentRoute:
    """Computes an approximate optimal spraying route connecting diseased polygons."""

    centroids = [poly.centroid for poly in polygons]
    if not centroids:
        return TreatmentRoute(order=[], waypoints=[start], total_length=0.0, segments=[])

    points = [start] + [(c.x, c.y) for c in centroids]
    graph = _build_complete_graph(points)

    tsp_order = nx.approximation.traveling_salesman_problem(graph, nodes=list(range(len(points))), cycle=False)

    waypoints = [points[idx] for idx in tsp_order]
    segments: List[RouteSegment] = []
    total_length = 0.0
    for start_idx, end_idx in zip(tsp_order[:-1], tsp_order[1:]):
        start_point = points[start_idx]
        end_point = points[end_idx]
        length = hypot(start_point[0] - end_point[0], start_point[1] - end_point[1])
        segments.append(RouteSegment(start=start_point, end=end_point, length=length))
        total_length += length

    order_labels = ["start" if idx == 0 else f"polygon_{idx}" for idx in tsp_order]
    return TreatmentRoute(order=order_labels, waypoints=waypoints, total_length=total_length, segments=segments)
