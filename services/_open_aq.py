# OpenAQ docs at: https://api.openaq.org/docs

from datetime import datetime, timedelta

import osmnx as ox
import requests
from dotenv import load_dotenv
import os

load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
CITY_OSMID = "R10108023"

# get our target city of choice, Bhubaneswar in our case
city = ox.geocoder.geocode_to_gdf(CITY_OSMID, by_osmid=True)
print(f"City name: {city.display_name.iloc[0]}")

l_lat, l_lon = city.bbox_south.iloc[0], city.bbox_west.iloc[0]
r_lat, r_lon = city.bbox_north.iloc[0], city.bbox_east.iloc[0]

r = requests.get(
    "https://api.openaq.org/v3/locations",
    params={"bbox": ",".join(map(str, [l_lon, l_lat, r_lon, r_lat])), "limit": 1000},
    headers={"X-API-Key": OPENAQ_API_KEY},
)

location_data = r.json()

# NOTE: The analysis below takes a very simplistic approach and as a result,
# a very limited amount of data into consideration, for representation and 
# basic prediction/calculation purposes
# a lot of the meta information is skipped via the various methods and viltering going on,
# but they can be added later on if needed (highly dubious xD)

# location_data.keys() -> dict_keys(['meta', 'results'])
# location_data['results'] is a list that contains all AQI stations for that location

# each element in that list is a dict, for that particular station
# location_data['results'][-1].keys() -> dict_keys(['id', 'name', 'locality', 'timezone', 'country', 'owner', 'provider', 'isMobile', 'isMonitor', 'instruments', 'sensors', 'coordinates', 'licenses', 'bounds', 'distance', 'datetimeFirst', 'datetimeLast'])

# so for each station, we only care about
# 'id, 'name', 'owner', 'sensors' and 'bounds'
# 'id' is the station id
# 'sensors' will give a list, each of whose elements is a dict with keys -> dict_keys(['id', 'name', 'parameter']). 'id' is the sensor id

aqi_stations = location_data["results"]
_aqi_stations = {} # station_id: {station details}

for station in aqi_stations:
    sensors = {}
    sensors_by_name = {}

    for sensor in station["sensors"]:
        sensors[sensor["id"]] = {
            "display_name": sensor["parameter"]["displayName"],
            "units": sensor["parameter"]["units"],
        }
        sensors_by_name[sensor["parameter"]["displayName"]] = {
            "id": sensor["id"],
            "units": sensor["parameter"]["units"],
        }

    _aqi_stations[station["id"]] = {
        "name": station["name"],
        "owner": station["owner"]["name"],
        "bbox_bounds": station["bounds"],
        "sensors": sensors,
        "sensors_by_name": sensors_by_name
    }

    # print(f"sensors: {[sensors[_]['display_name'] for _ in sensors]}\n")


def get_latest_readings_from_station(station_id: str, limit: int = 100) -> dict: # sensor_id: readings
    readings = requests.get(
        f"https://api.openaq.org/v3/locations/{station_id}/latest",
        params={"limit": limit},
        headers={"X-API-Key": OPENAQ_API_KEY},
    ).json()["results"]

    _readings = {}
    for reading in readings:
        sensor_id = reading["sensorsId"]
        _station_sensor = _aqi_stations[station_id]["sensors"][sensor_id]

        _readings[sensor_id] = {
            "value": reading["value"],
            "display_name": _station_sensor["display_name"],
            "units": _station_sensor["units"],
            "datetime": datetime.strptime(
                reading["datetime"]["utc"], r"%Y-%m-%dT%H:%M:%SZ"
            ),  # 2026-03-28T16:00:00Z
        }

    return _readings


def get_hourly_readings_from_sensor(sensor_id: str, hours: int = 24, limit: int = 100) -> list: # list of readings
    readings = requests.get(
        f"https://api.openaq.org/v3/sensors/{sensor_id}/hours",
        params={
            "limit": limit,
            "datetime_from": (datetime.now() - timedelta(hours=hours))
            .date()
            .isoformat(),
            "datetime_to": datetime.now().date().isoformat(),
        },
        headers={"X-API-Key": OPENAQ_API_KEY},
    ).json()["results"]

    _readings = []
    for reading in readings:
        _readings.append(
            {
                "value": reading["value"],
                "display_name": reading["parameter"]["name"],
                "units": reading["parameter"]["units"],
                "datetime": datetime.strptime(
                    reading["period"]["datetimeTo"]["utc"], r"%Y-%m-%dT%H:%M:%SZ"
                ),
            }
        )
    
    return _readings

# so now we have access to all the stations for a given location, which we can query by their station id
# and each station has a bunch of sensors, which we can query by their sensor id

# and now, after defining the above methods, we can either:
# 1. get the latest redings for all sensors in any station
# 2. get an arbitrary range of hourly readings for any specific sensor [note that each sensor is unique, in which, it depends on the station]


# for example, say we did all of this, and now on our homepage, we want to display a nice chart about the latest readings
# (last 24h) of the CO ppb censor at patia, or say in this example, across Bhubaneswar

for _station_id in _aqi_stations:
    station_sensors_by_id = _aqi_stations[_station_id]['sensors']
    station_sensors_by_name = _aqi_stations[_station_id]['sensors_by_name']

    print(f"\nFor station {_aqi_stations[_station_id]['name']}...")
    print(f"Avaliable sensors are: {station_sensors_by_name.keys()}")

    target_sensor_id = station_sensors_by_name[input("Enter sensor of choice: ").strip()]['id']
    print("Getting readings...")
    for reading in get_hourly_readings_from_sensor(target_sensor_id):
        print(reading['datetime'].isoformat(), reading['value'], reading['units'])

