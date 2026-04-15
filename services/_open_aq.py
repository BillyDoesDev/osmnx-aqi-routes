# OpenAQ docs at: https://api.openaq.org/docs

from datetime import datetime, timedelta
from typing import List, Tuple

import osmnx as ox
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
import os
# from graph_service import CITY_OSMID
from collections import defaultdict
from services.model import N_LAGS


load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")


class SensorRef(BaseModel):
    """Lightweight ref used in sensors_by_name lookup (name -> id + units)."""

    id: int
    units: str


class StationSensor(BaseModel):
    """A single sensor entry on a station (id -> display_name + units)."""

    id: int
    display_name: str
    units: str


class Station(BaseModel):
    name: str
    owner: str
    bbox_bounds: list[float]  # [w, s, e, n]
    sensors: dict[int, StationSensor]  # sensor_id -> StationSensor
    sensors_by_name: dict[str, SensorRef]  # display_name -> SensorRef


class CityAQI(BaseModel):
    """Top-level object for a city - holds all its AQI stations."""

    city_name: str
    stations: dict[int, Station]  # station_id -> Station


class Reading(BaseModel):
    value: float
    display_name: str
    units: str
    datetime: datetime
    datetime_as_str: str


# get our target city of choice, Bhubaneswar in our case
# NOTE: The analysis below takes a very simplistic approach and as a result,
# a very limited amount of data into consideration, for representation and
# basic prediction/calculation purposes
# a lot of the meta information is skipped via the various methods and filtering going on,
# but they can be added later on if needed (highly dubious xD)

# location_data.keys() -> dict_keys(['meta', 'results'])
# location_data['results'] is a list that contains all AQI stations for that location

# each element in that list is a dict, for that particular station
# location_data['results'][-1].keys() -> dict_keys(['id', 'name', 'locality', 'timezone', 'country', 'owner', 'provider', 'isMobile', 'isMonitor', 'instruments', 'sensors', 'coordinates', 'licenses', 'bounds', 'distance', 'datetimeFirst', 'datetimeLast'])

# so for each station, we only care about
# 'id, 'name', 'owner', 'sensors' and 'bounds'
# 'id' is the station id
# 'sensors' will give a list, each of whose elements is a dict with keys -> dict_keys(['id', 'name', 'parameter']). 'id' is the sensor id


def fetch_city_aqi(osmid: str) -> CityAQI:
    """Fetch all AQI stations for a city by its OSM ID."""
    city = ox.geocoder.geocode_to_gdf(osmid, by_osmid=True)
    city_name = city.display_name.iloc[0]
    print(f"City: {city_name}")

    l_lat, l_lon = city.bbox_south.iloc[0], city.bbox_west.iloc[0]
    r_lat, r_lon = city.bbox_north.iloc[0], city.bbox_east.iloc[0]

    results = requests.get(
        "https://api.openaq.org/v3/locations",
        params={
            "bbox": ",".join(map(str, [l_lon, l_lat, r_lon, r_lat])),
            "limit": 1000,
        },
        headers={"X-API-Key": OPENAQ_API_KEY},
    ).json()["results"]

    stations: dict[int, Station] = {}
    for station in results:
        sensors: dict[int, StationSensor] = {}
        sensors_by_name: dict[str, SensorRef] = {}

        for sensor in station["sensors"]:
            param = sensor["parameter"]
            sensors[sensor["id"]] = StationSensor(
                id=sensor["id"],
                display_name=param["displayName"],
                units=param["units"],
            )
            sensors_by_name[param["displayName"]] = SensorRef(
                id=sensor["id"],
                units=param["units"],
            )

        stations[station["id"]] = Station(
            name=station["name"],
            owner=station["owner"]["name"],
            bbox_bounds=station["bounds"],
            sensors=sensors,
            sensors_by_name=sensors_by_name,
        )

    return CityAQI(city_name=city_name, stations=stations)


# API HELPERS


def get_latest_readings_from_station(
    city: CityAQI, station_id: int, limit: int = 100
) -> dict[int, Reading]:
    """Fetch the latest reading for every sensor at a station."""
    results = requests.get(
        f"https://api.openaq.org/v3/locations/{station_id}/latest",
        params={"limit": limit},
        headers={"X-API-Key": OPENAQ_API_KEY},
    ).json()["results"]

    readings: dict[int, Reading] = {}
    for result in results:
        sensor_id = result["sensorsId"]
        sensor = city.stations[station_id].sensors[sensor_id]
        readings[sensor_id] = Reading(
            value=result["value"],
            display_name=sensor.display_name,
            units=sensor.units,
            datetime=datetime.strptime(
                result["datetime"]["utc"], r"%Y-%m-%dT%H:%M:%SZ"
            ),
        )

    return readings


def get_hourly_readings_from_sensor(
    sensor_id: int, hours: int = 24, limit: int = 100
) -> list[Reading]:
    """Fetch hourly readings for a specific sensor over the last N hours."""
    results = requests.get(
        f"https://api.openaq.org/v3/sensors/{sensor_id}/hours",
        params={
            "limit": limit,
            "datetime_from": (datetime.now() - timedelta(hours=hours))
            .isoformat(),
            "datetime_to": datetime.now().isoformat(),
        },
        headers={"X-API-Key": OPENAQ_API_KEY},
    ).json()["results"]

    return [
        Reading(
            value=result["value"],
            display_name=result["parameter"]["name"],
            units=result["parameter"]["units"],
            datetime=datetime.strptime(
                result["period"]["datetimeTo"]["utc"], r"%Y-%m-%dT%H:%M:%SZ"
            ),
            datetime_as_str=result["period"]["datetimeTo"]["utc"],
        )
        for result in results
    ]


# so now we have access to all the stations for a given location, which we can query by their station id
# and each station has a bunch of sensors, which we can query by their sensor id

# and now, after defining the above methods, we can either:
# 1. get the latest redings for all sensors in any station
# 2. get an arbitrary range of hourly readings for any specific sensor [note that each sensor is unique, in which, it depends on the station]


def get_recent_station_readings(city:CityAQI, N_LAGS:int=N_LAGS) -> List[Tuple[int, List[dict]]]:
    # maps to FEATURES, as trained in the model
    _keys = [('PM10', 'pm10'), ('PM2.5', 'pm25'), ('RH', 'relativehumidity'), ('Temperature (C)', 'temperature'), ('Wind direction', 'wind_direction'), ('Wind speed', 'wind_speed')]
    recent_station_readings = []

    for station_id, station in city.stations.items():
        _num_readings = 0
        _sensor_readings = defaultdict(list)

        for _k, needed_sensor in _keys:
            needed_sensor_id = station.sensors_by_name[_k].id
            n_lags_readings = get_hourly_readings_from_sensor(sensor_id=needed_sensor_id, hours=N_LAGS*2)
            _num_readings = len(n_lags_readings)
            
            for reading in n_lags_readings:
                _sensor_readings[needed_sensor].append((reading.datetime_as_str, reading.value))
        
        recent_readings = []
        for i in range(_num_readings):
            recent_reading = {}
            for _, key in _keys:
                # print(key, _sensor_readings[key][i])
                recent_reading[key] = _sensor_readings[key][i][1]
            recent_reading["datetime"] = _sensor_readings[key][i][0]
            recent_readings.append(recent_reading)
        
        recent_station_readings.append( (station_id, recent_readings) )

    return recent_station_readings


# EXAMPLE USAGE
# say we did all of this, and now on our homepage, we want to display a nice chart about the latest readings
# (last 4h) of the CO ppb censor at patia, or say in this example, across Bhubaneswar

if __name__ == "__main__":
    pass
    # city = fetch_city_aqi(CITY_OSMID)

    # for station_id, station in city.stations.items():
    #     print(f"\nFor station {station.name}...")
    #     print(f"Available sensors: {list(station.sensors_by_name.keys())}")

    #     sensor_name = input("Enter sensor of choice: ").strip()
    #     target_sensor_id = station.sensors_by_name[sensor_name].id

    #     print("Getting readings...")
    #     for reading in get_hourly_readings_from_sensor(target_sensor_id):
    #         print(reading.datetime.isoformat(), reading.value, reading.units)
    