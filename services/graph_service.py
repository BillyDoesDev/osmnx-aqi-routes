from folium.plugins import Fullscreen
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry import shape, Point, polygon
from typing import Tuple
import folium
import json
import logging
import osmnx as ox
from services.routing_service import RouteResult, compute_routes, get_directions_for_route
# from services._open_aq import fetch_city_aqi, get_recent_station_readings
# from services.model import predict_pm25
import networkx as nx

city = None
_boundary_geojson = None  # raw GeoJSON geometry dict
_boundary_shape = None  # shapely shape for point-in-polygon
OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"

CITY_OSMID = "R10108023"

def get_city_osmid():
    return CITY_OSMID

# _city = fetch_city_aqi(CITY_OSMID)
# station_preds = {
#     station_id: predict_pm25(station_id, station_readings)
#     for station_id, station_readings in get_recent_station_readings(_city)
# }

def get_boundary() -> Tuple[GeoDataFrame, str, polygon.Polygon]:
    """Fetch Bhubaneswar's actual political boundary polygon."""
    global city, _boundary_geojson, _boundary_shape

    # https://nominatim.openstreetmap.org/ui/search.html?q=R10108023
    city = ox.geocoder.geocode_to_gdf(CITY_OSMID, by_osmid=True)

    _boundary_geojson = city.to_json()
    _boundary_shape = shape(json.loads(_boundary_geojson)["features"][0]["geometry"])

    return city, _boundary_geojson, _boundary_shape


def generate_base_map(fresh_map=True) -> folium.Map:
    city, geojson, _ = get_boundary()

    m = folium.Map(
        location=[city.loc[0, "lat"], city.loc[0, "lon"]],
        zoom_start=13,
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    )
    Fullscreen().add_to(m)

    # get bounding rectangle
    # lat, lon = city.lat.iloc[0], city.lon.iloc[0]
    # l_lat, l_lon = city.bbox_south.iloc[0], city.bbox_west.iloc[0]
    # r_lat, r_lon = city.bbox_north.iloc[0], city.bbox_east.iloc[0]

    # m = folium.Map(location=[lat, lon], zoom_start=13)
    # folium.Marker(location=[lat, lon]).add_to(m)
    # folium.Rectangle(
    #     bounds=[[l_lat, l_lon], [r_lat, r_lon]],
    # ).add_to(m)

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
    if fresh_map: folium.LayerControl().add_to(m)

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


def render_routes_on_map(
    m: folium.Map,
    fastest: RouteResult,
    optimal: RouteResult,
    fresh_map=True,
) -> folium.Map:

    # Fastest route — grey, slightly transparent
    fastest_route_layer = folium.FeatureGroup(name="Fastest Route")
    folium.PolyLine(
        fastest.coords,
        color="#0077ff",
        weight=4,
        opacity=0.6,
        tooltip=f"Fastest — {round(fastest.total_time / 60, 1)} min, avg PM2.5: {round(fastest.mean_pm25)} µg/m³",
    ).add_to(fastest_route_layer)
    fastest_route_layer.add_to(m)

    # Optimal route — blue, full opacity
    optimal_route_layer = folium.FeatureGroup(name="Optimal Route")
    folium.PolyLine(
        optimal.coords,
        color="#1B6D56",
        weight=5,
        opacity=0.9,
        tooltip=f"Optimal — {round(optimal.total_time / 60, 1)} min, avg PM2.5: {round(optimal.mean_pm25)} µg/m³",
    ).add_to(optimal_route_layer)
    optimal_route_layer.add_to(m)

    if fresh_map: folium.LayerControl().add_to(m)
    return m


def generate_route_map(start_place: str, end_place: str, station_preds: dict[int, float], _G: nx.MultiDiGraph, alpha:int=0):
    start_lat, start_lon = geocode_within_boundary(start_place)
    end_lat, end_lon = geocode_within_boundary(end_place)

    logging.debug(f"{start_lat, start_lon, end_lat, end_lon = }")

    m = generate_base_map(fresh_map=False)

    fastest, optimal = compute_routes((start_lat, start_lon), (end_lat, end_lon), station_preds, _G, alpha)
    m = render_routes_on_map(m, fastest, optimal, fresh_map=False)

    fastest_steps = get_directions_for_route(_G, fastest.node_sequence)
    optimal_steps = get_directions_for_route(_G, optimal.node_sequence)

    # km, min
    fastest_distance, fastest_time = fastest.total_distance/1e3, fastest.total_time/(60)
    optimal_distance, optimal_time = optimal.total_distance/1e3, optimal.total_time/(60)

    place_markers_layer = folium.FeatureGroup(name="Place Markers")
    folium.Marker(
        location=(start_lat, start_lon),
        icon=folium.Icon(color="green"),
        tooltip=f"Start: {start_place}",
    ).add_to(place_markers_layer)

    folium.Marker(
        location=(end_lat, end_lon),
        icon=folium.Icon(color="red"),
        tooltip=f"End: {end_place}",
    ).add_to(place_markers_layer)
    place_markers_layer.add_to(m)

    folium.LayerControl().add_to(m)

    return m, (fastest_steps, optimal_steps), (fastest_distance, optimal_distance), (fastest_time, optimal_time), (fastest.mean_pm25, optimal.mean_pm25)
