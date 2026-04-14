# now let's try downloading all data from their s3 stores for us to analyse
# ref: https://docs.openaq.org/aws/quick-start

from datetime import datetime
import glob
import os
import subprocess

from graph_service import CITY_OSMID
from _open_aq import fetch_city_aqi


city = fetch_city_aqi(CITY_OSMID)
year = 2025
base_s3_command_args = ["aws", "s3", "cp", "--no-sign-request", "--recursive"]

while year <= datetime.now().year:
    for station_id, station in city.stations.items():
        print(f"[{year}] downloading data for: {station.name} [{station_id}]")
        _ = subprocess.run(
            base_s3_command_args
            + [
                f"s3://openaq-data-archive/records/csv.gz/locationid={station_id}/year={year}/",
                f"data-{year}-{station_id}",
            ]
        )
    year += 1

# extract the fownloaded files
print("Extracting files using gunzip...")
for root, dirs, files in os.walk("."):
    for file_path in glob.iglob(root + "/*.gz"):
        _ = subprocess.run(["gunzip", file_path])
