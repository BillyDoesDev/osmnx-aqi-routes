from folium.plugins import Fullscreen
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry import shape, Point, polygon
from typing import Tuple
import folium
import json
import logging
import osmnx as ox
import requests

city = None
_boundary_geojson = None  # raw GeoJSON geometry dict
_boundary_shape = None  # shapely shape for point-in-polygon
OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"


def get_boundary() -> Tuple[GeoDataFrame, str, polygon.Polygon]:
    """Fetch Bhubaneswar's actual political boundary polygon."""
    global city, _boundary_geojson, _boundary_shape

    # https://nominatim.openstreetmap.org/ui/search.html?q=R10108023
    city = ox.geocoder.geocode_to_gdf("R10108023", by_osmid=True)

    _boundary_geojson = city.to_json()
    _boundary_shape = shape(json.loads(_boundary_geojson)["features"][0]["geometry"])

    return city, _boundary_geojson, _boundary_shape


def generate_base_map() -> folium.Map:
    city, geojson, _ = get_boundary()

    m = folium.Map(
        location=[city.loc[0, "lat"], city.loc[0, "lon"]],
        zoom_start=13,
        tiles="OpenStreetMap",
    )
    Fullscreen().add_to(m)

    folium.GeoJson(
        geojson,
        name="Supported routing area",
        # tooltip="Supported routing area",
        style_function=lambda _: {
            "fillColor": "#4488ff",
            "fillOpacity": 0.08,
            "color": "#4488ff",
            "weight": 2,
            "dashArray": "6 4",
            "opacity": 0.5,
        },
    ).add_to(m)
    folium.LayerControl().add_to(m)

    return m


def point_in_boundary(lat, lon) -> bool:
    """Return True if (lat, lon) is within defined boundary."""
    _, _, shp = get_boundary()
    if shp is not None:
        return shp.contains(Point(lon, lat))


def geocode_within_boundary(place: str):
    """Geocode a place, verify it's inside defined boundary."""
    lat, lon = ox.geocode(place)

    if not point_in_boundary(lat, lon):
        raise ValueError(f"'{place}' appears to be outside the defined boundary.")

    return lat, lon


def get_osrm_route(start_lat, start_lon, end_lat, end_lon):
    coords = f"{start_lon},{start_lat};{end_lon},{end_lat}"
    resp = requests.get(
        f"{OSRM_BASE}/{coords}",
        params={"overview": "full", "geometries": "geojson", "steps": "true"},
        timeout=15,
    )
    data = resp.json()

    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError("OSRM could not find a route between these locations.")

    route = data["routes"][0]
    leg = route["legs"][0]

    coords_latlon = [(pt[1], pt[0]) for pt in route["geometry"]["coordinates"]]

    steps = []
    for step in leg["steps"]:
        maneuver = step.get("maneuver", {})
        mtype = maneuver.get("type", "")
        modifier = maneuver.get("modifier", "")
        name = step.get("name", "").strip() or "unnamed road"
        distance = step.get("distance", 0)

        if mtype == "depart":
            instruction = (
                f"Head {modifier} on {name}" if modifier else f"Start on {name}"
            )
        elif mtype == "arrive":
            instruction = "Arrive at destination"
        elif mtype == "turn":
            instruction = f"Turn {modifier} onto {name}"
        elif mtype == "new name":
            instruction = f"Continue onto {name}"
        elif mtype == "merge":
            instruction = f"Merge onto {name}"
        elif mtype in ("on ramp", "off ramp"):
            instruction = f"Take the {'ramp' if 'on' in mtype else 'exit'} onto {name}"
        elif mtype == "fork":
            instruction = f"Keep {modifier} at the fork onto {name}"
        elif mtype in ("roundabout", "rotary"):
            exit_num = maneuver.get("exit", "")
            instruction = f"At the roundabout, take exit {exit_num} onto {name}"
        elif mtype == "end of road":
            instruction = f"At the end of the road, turn {modifier} onto {name}"
        else:
            instruction = f"Continue on {name}"

        steps.append(
            {
                "instruction": instruction,
                "distance": round(distance),
                "type": mtype,
            }
        )

    return (
        coords_latlon,
        steps,
        round(route["distance"] / 1000, 2),
        round(route["duration"] / 60, 1),
    )


def generate_route_map(start_place: str, end_place: str):
    start_lat, start_lon = geocode_within_boundary(start_place)
    end_lat, end_lon = geocode_within_boundary(end_place)

    logging.debug(f"{start_lat, start_lon, end_lat, end_lon = }")

    route_coords, steps, dist_km, time_min = get_osrm_route(
        start_lat, start_lon, end_lat, end_lon
    )

    m = generate_base_map()

    folium.PolyLine(route_coords, color="#0077ff", weight=5, opacity=0.9).add_to(m)

    folium.Marker(
        location=(start_lat, start_lon),
        icon=folium.Icon(color="green"),
        tooltip=f"Start: {start_place}",
    ).add_to(m)

    folium.Marker(
        location=(end_lat, end_lon),
        icon=folium.Icon(color="red"),
        tooltip=f"End: {end_place}",
    ).add_to(m)

    return m, steps, dist_km, time_min
