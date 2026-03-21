import folium
from folium.plugins import Fullscreen
from shapely.geometry import shape, Point
import json
import osmnx as ox

city = None


def get_boundary():
    global city

    # https://nominatim.openstreetmap.org/ui/search.html?q=R10108023
    city = ox.geocoder.geocode_to_gdf("R10108023", by_osmid=True)

    _boundary_geojson = city.to_json()
    _boundary_shape = shape(json.loads(_boundary_geojson)["features"][0]["geometry"])

    return _boundary_geojson, _boundary_shape


def generate_base_map():
    geojson, _ = get_boundary()

    m = folium.Map(
        location=[city.lat[0], city.lon[0]], zoom_start=13, tiles="OpenStreetMap"
    )
    Fullscreen().add_to(m)

    folium.GeoJson(
        geojson,
        name="Supported routing area",
        tooltip="Supported routing area",
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
