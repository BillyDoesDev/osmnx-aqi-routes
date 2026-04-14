# routing_service.py
# Computes two routes:
#   1. Fastest  -- standard travel time via OSRM (clean directions)
#   2. Optimal  -- pollution + time weighted via OSMnx + our PM2.5 model
#
# Modular: swap out get_travel_time_weight() when live traffic is available.

from __future__ import annotations

import networkx as nx
import osmnx as ox
import numpy as np
from dataclasses import dataclass

import requests

from model import interpolate_pm25_for_nodes, predict_pm25
from graph_service import CITY_OSMID
from _open_aq import get_recent_station_readings, fetch_city_aqi


DEFAULT_ALPHA = 0.5
_graph: nx.MultiDiGraph | None = None
_city = ox.geocoder.geocode_to_gdf(CITY_OSMID, by_osmid=True)

print("Loading OSMnx graph for Bhubaneswar...")
G = ox.graph_from_place(_city.display_name.iloc[0], network_type="drive")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
_graph = G
print(f"Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges.")


def get_graph() -> nx.MultiDiGraph:
    global _graph
    if _graph is None:
        print("Loading OSMnx graph for Bhubaneswar...")
        G = ox.graph_from_place(_city.display_name.iloc[0], network_type="drive")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        _graph = G
        print(f"Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges.")
    return _graph


# edge weight hooks, replace as needed
def get_travel_time_weight(G: nx.MultiDiGraph, u: int, v: int, data: dict) -> float:
    """
    Travel time in seconds for edge (u, v).
    Currently uses OSMnx static speed estimates.
    TODO: replace with live traffic speed when available -- this is the only
          function you need to change when adding a traffic data source.
    """
    # _ = data.get("travel_time", data.get("length", 1) / 6.944)  # fallback: 25 km/h
    # print(f"{data.get("length", 1) / 6.944, data.get("travel_time", None), _ = }")
    return data.get("length", 1) / 6.944 # fallback: 25 km/h


# update as you get better data sources
ROAD_POLLUTION_MULTIPLIER = {
    "motorway": 2.0,
    "trunk": 1.8,
    "primary": 1.5,
    "secondary": 1.2,
    "tertiary": 1.0,
    "residential": 0.7,
    "unclassified": 0.9,
}


def get_pollution_weight(
    u: int,
    v: int,
    data,
    node_pm25: dict[int, float],
) -> float:
    """
    PM2.5 exposure for edge (u, v), estimated as the average of its
    two endpoint node predictions.
    """
    pm_u = node_pm25.get(u, 0.0)
    pm_v = node_pm25.get(v, 0.0)
    base = (pm_u + pm_v) / 2.0
    highway = data.get("highway", "unclassified")
    if isinstance(highway, list):
        highway = highway[0]
    multiplier = ROAD_POLLUTION_MULTIPLIER.get(highway, 1.0)
    return base * multiplier


# composite weight
def build_composite_graph(
    G: nx.MultiDiGraph,
    node_pm25: dict[int, float],
    alpha: float = DEFAULT_ALPHA,
) -> nx.MultiDiGraph:
    """
    Attach a 'composite_weight' attribute to every edge:

        composite = alpha x norm(travel_time) + (1 - alpha) x norm(pm25)

    Both components are normalized to [0, 1] across all edges so neither
    dominates by scale. Alpha is the time-vs-pollution tradeoff slider.
    """
    # Collect raw values for normalization
    travel_times = []
    pollution_vals = []

    for u, v, data in G.edges(data=True):
        travel_times.append(get_travel_time_weight(G, u, v, data))
        pollution_vals.append(get_pollution_weight(u, v, data, node_pm25))

    tt_arr = np.array(travel_times, dtype=float)
    pol_arr = np.array(pollution_vals, dtype=float)

    # Min-max normalization -- safe against zero-range edge case
    def norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    tt_norm = norm(tt_arr)
    pol_norm = norm(pol_arr)

    G = G.copy()
    for i, (u, v, k) in enumerate(G.edges(keys=True)):
        G[u][v][k]["composite_weight"] = alpha * tt_norm[i] + (1 - alpha) * pol_norm[i]

    return G


# route result
@dataclass
class RouteResult:
    node_sequence: list[int]
    coords: list[tuple[float, float]]   # (lat, lon) for map rendering
    total_distance: float               # metres
    total_time: float                   # seconds
    mean_pm25: float                    # average predicted PM2.5 along route
    edge_pm25: list[float]              # per-edge PM2.5 for timeline display


def _extract_route(
    G: nx.MultiDiGraph,
    route: list[int],
    node_pm25: dict[int, float],
) -> RouteResult:
    coords, distances, times, pm25_vals = [], [], [], []

    for i, node in enumerate(route):
        coords.append((G.nodes[node]["y"], G.nodes[node]["x"]))
        if i < len(route) - 1:
            u, v = route[i], route[i + 1]
            data = G.get_edge_data(u, v, 0) or {}
            distances.append(data.get("length", 0))
            times.append(get_travel_time_weight(G, u, v, data))
            pm25_vals.append(get_pollution_weight(u, v, data, node_pm25))

    return RouteResult(
        node_sequence=route,
        coords=coords,
        total_distance=sum(distances),
        total_time=sum(times),
        mean_pm25=float(np.mean(pm25_vals)) if pm25_vals else 0.0,
        edge_pm25=pm25_vals,
    )


# main entry
def compute_routes(
    start_coords: tuple[float, float],
    end_coords: tuple[float, float],
    station_predictions: dict[int, float],  # {station_id: predicted_pm25}
    alpha: float = DEFAULT_ALPHA,
) -> tuple[RouteResult, RouteResult]:
    """
    Compute and return (fastest_route, optimal_route).

    fastest_route:  shortest travel time, ignores pollution
    optimal_route:  composite time + pollution score at given alpha

    station_predictions: output of predict_pm25() per station --
        {station_id: pm25_value}. Pass the 30-min-ahead predictions
        so the route reflects expected conditions when you arrive.

    Returns two RouteResult objects ready for map rendering and
    pollution timeline display.
    """
    G = get_graph()

    orig = ox.nearest_nodes(G, start_coords[1], start_coords[0])
    dest = ox.nearest_nodes(G, end_coords[1], end_coords[0])

    # Interpolate PM2.5 to every graph node from station predictions
    node_coords = [(n, G.nodes[n]["y"], G.nodes[n]["x"]) for n in G.nodes]
    node_pm25 = interpolate_pm25_for_nodes(node_coords, station_predictions)

    # pm25_values = list(node_pm25.values())
    # print(np.min(pm25_values), np.max(pm25_values), np.std(pm25_values))

    # ── Fastest route (travel time only) ──
    fastest_nodes = nx.shortest_path(G, orig, dest, weight="travel_time")
    fastest = _extract_route(G, fastest_nodes, node_pm25)

    # ── Optimal route (composite weight) ──
    G_composite = build_composite_graph(G, node_pm25, alpha)
    optimal_nodes = nx.shortest_path(G_composite, orig, dest, weight="composite_weight")
    optimal = _extract_route(G_composite, optimal_nodes, node_pm25)

    return fastest, optimal


# def build_pollution_timeline(
#     route: RouteResult,
#     pm25_sequence: list[dict],  # output of predict_pm25_sequence()
#     interval_minutes: int = 15,
# ) -> list[dict]:
#     """
#     Map a station's PM2.5 forecast sequence onto 15-minute waypoints
#     along the route for display as a timeline chart.

#     pm25_sequence: output of predict_pm25_sequence(n_steps=10) --
#         [{"step": 1, "minutes": 15, "pm25": ..., "datetime": ...}, ...]

#     Returns a list of dicts suitable for charting:
#         [{"minutes": 0,  "pm25": 120.5, "datetime": "...", "label": "Now"},
#          {"minutes": 15, "pm25": 118.2, "datetime": "...", "label": "+15 min"},
#          ...]
#     """
#     # Current conditions (step 0) -- use route mean as the "now" baseline
#     timeline = [
#         {
#             "minutes": 0,
#             "pm25": round(route.mean_pm25, 2),
#             "datetime": None,
#             "label": "Now",
#         }
#     ]

#     for entry in pm25_sequence:
#         timeline.append(
#             {
#                 "minutes": entry["minutes"],
#                 "pm25": entry["pm25"],
#                 "datetime": entry["datetime"],
#                 "label": f"+{entry['minutes']} min",
#             }
#         )

#     return timeline


def get_decision_nodes(G: nx.MultiDiGraph, node_sequence: list[int]) -> list[tuple[float, float]]:
    """Extract only nodes where the road changes direction or name."""
    if len(node_sequence) < 2:
        return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_sequence]

    waypoints = [node_sequence[0]]  # always include start

    for i in range(1, len(node_sequence) - 1):
        prev, curr, next_ = node_sequence[i-1], node_sequence[i], node_sequence[i+1]
        
        # Check if road name changes at this node
        edge_in  = G.get_edge_data(prev, curr,  0) or {}
        edge_out = G.get_edge_data(curr, next_, 0) or {}
        name_in  = edge_in.get("name",  "")
        name_out = edge_out.get("name", "")
        
        # Check if road type changes (e.g. primary -> residential)
        hw_in  = edge_in.get("highway",  "")
        hw_out = edge_out.get("highway", "")

        if name_in != name_out or hw_in != hw_out:
            waypoints.append(curr)

    waypoints.append(node_sequence[-1])  # always include end
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in waypoints]


def get_directions_for_route(G: nx.MultiDiGraph, node_sequence: list[int]) -> list[dict]:
    """
    Given a sequence of (lat, lon) coords from OSMnx routing,
    snap to OSRM via /match and get back turn-by-turn directions.
    """
    waypoints = get_decision_nodes(G, node_sequence)
    # print(len(waypoints))

    # Safety cap — shouldn't hit this often now
    MAX_WAYPOINTS = 10
    if len(waypoints) > MAX_WAYPOINTS:
        indices   = np.linspace(0, len(waypoints) - 1, MAX_WAYPOINTS, dtype=int)
        waypoints = [waypoints[i] for i in indices]

    coords_str = ";".join(f"{lon},{lat}" for lat, lon in waypoints)


    resp = requests.get(
        f"http://router.project-osrm.org/match/v1/car/{coords_str}",
        params={
            "steps":     "true",
            "geometries": "polyline",
            "overview":  "false",
            # "ttype":     "duration",  # match by duration, not distance
        },
        timeout=15,
    )
    data = resp.json()

    if data.get("code") != "Ok" or not data.get("matchings"):
        return []

    steps = []
    for matching in data["matchings"]:
        for leg in matching["legs"]:
            for step in leg["steps"]:
                maneuver = step.get("maneuver", {})
                mtype    = maneuver.get("type", "")
                modifier = maneuver.get("modifier", "")
                name     = step.get("name", "").strip() or "unnamed road"
                distance = step.get("distance", 0)

                if mtype == "depart":
                    instruction = f"Head {modifier} on {name}" if modifier else f"Start on {name}"
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

                steps.append({
                    "instruction": instruction,
                    "distance":    round(distance),
                    "type":        mtype,
                })

    return steps


if __name__ == "__main__":
    city = fetch_city_aqi(CITY_OSMID)
    station_preds = {
        station_id: predict_pm25(station_id, station_readings)
        for station_id, station_readings in get_recent_station_readings(city)
    }

    _orig = ox.geocoder.geocode_to_gdf("kiit campus 6")
    start_coords = _orig["lat"].iloc[0], _orig["lon"].iloc[0]

    _dest = ox.geocoder.geocode_to_gdf("Master Canteen")
    end_coords = _dest["lat"].iloc[0], _dest["lon"].iloc[0]

    fastest, optimal = compute_routes(start_coords, end_coords, station_preds, alpha=0)
    print(f"{fastest.mean_pm25, optimal.mean_pm25 = }")
    print(f"{fastest.total_distance/1e3, fastest.total_time/(60) = }")
    print(f"{optimal.total_distance/1e3, optimal.total_time/(60) = }")

    print("getting directions for optimal route now...")
    optimal_steps = get_directions_for_route(_graph, optimal.node_sequence)
    for step in optimal_steps:
        print(step)

