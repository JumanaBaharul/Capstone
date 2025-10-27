"""Mapping utilities for treatment plan visualisation."""

from __future__ import annotations

from typing import Iterable, Tuple

import folium
from branca.colormap import linear
from math import cos, radians

from .config import VisualizationConfig
from .inference import PredictionResult
from .route_planner import TreatmentRoute


def _project_to_latlon(x: float, y: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    meters_per_degree_lat = 111_320
    lat = origin_lat + y / meters_per_degree_lat
    meters_per_degree_lon = 111_320 * max(0.1, abs(cos(radians(origin_lat))))
    lon = origin_lon + x / meters_per_degree_lon
    return lat, lon


def create_treatment_map(
    predictions: Iterable[PredictionResult],
    route: TreatmentRoute,
    config: VisualizationConfig,
    origin_lat: float = 0.0,
    origin_lon: float = 0.0,
) -> folium.Map:
    """Exports an interactive folium map summarising predictions and routes."""

    config.map_output.parent.mkdir(parents=True, exist_ok=True)
    fmap = folium.Map(location=[origin_lat, origin_lon], zoom_start=config.zoom_start, control_scale=True)

    colormap_factory = getattr(linear, config.colormap, linear.YlOrRd_09)
    colormap = colormap_factory.scale(0, 1)
    colormap.add_to(fmap)

    for prediction in predictions:
        for polygon in prediction.polygons:
            coordinates = [
                _project_to_latlon(x, y, origin_lat, origin_lon) for x, y in polygon.exterior.coords
            ]
            folium.Polygon(
                locations=[(lat, lon) for lat, lon in coordinates],
                color=colormap(prediction.probability_map.max()),
                fill=True,
                fill_color=colormap(prediction.probability_map.max()),
                weight=2,
                popup=f"{prediction.sample_id}",
            ).add_to(fmap)

    for segment in route.segments:
        start_latlon = _project_to_latlon(segment.start[0], segment.start[1], origin_lat, origin_lon)
        end_latlon = _project_to_latlon(segment.end[0], segment.end[1], origin_lat, origin_lon)
        folium.PolyLine(locations=[start_latlon, end_latlon], color="blue", weight=3).add_to(fmap)

    fmap.save(config.map_output)
    return fmap
