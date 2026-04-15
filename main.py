from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from services._open_aq import fetch_city_aqi, get_recent_station_readings
from services.graph_service import generate_base_map, get_boundary, generate_route_map, get_city_osmid
import logging

from services.model import predict_pm25
from services.routing_service import get_graph


logging.basicConfig(
    format="[{levelname}] {asctime}: {message}", style="{", level=logging.DEBUG
)

city = None
_boundary_geojson = None
_boundary_shape = None

_city = None
station_preds = None
_G = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.debug("Setting up lifespan...")
    global city, _boundary_geojson, _boundary_shape, _city, station_preds, _G
    city, _boundary_geojson, _boundary_shape = get_boundary()

    _city = fetch_city_aqi(get_city_osmid())
    station_preds = {
        station_id: predict_pm25(station_id, station_readings)
        for station_id, station_readings in get_recent_station_readings(_city)
    }

    _G = get_graph()

    yield
    # Clean up code here


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    m = generate_base_map()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "map_html": m._repr_html_(),
        },
    )


@app.get("/city-metadata", response_class=JSONResponse)
async def get_bbox_bounds():
    west, south, east, north = [
        _.loc[0]
        for _ in (city.bbox_west, city.bbox_south, city.bbox_east, city.bbox_north)
    ]
    data = {
        "bbox-bounds": {"west": west, "south": south, "east": east, "north": north},
        "name": city.display_name.loc[0],
    }
    return JSONResponse(content=data)


@app.get("/route", response_class=HTMLResponse)
async def route(
    request: Request,
    start: str = Query(...),
    end: str = Query(...),
    alpha_input: float = Query(...),
):
    logging.debug(f"[Alpha: {alpha_input}] Got /route: {start}->{end}")
    try:
        m, (fastest_steps, optimal_steps), (fastest_dist, optimal_dist), (fastest_time, optimal_time), (fastest_mean_pm25, optimal_mean_pm25) = generate_route_map(start, end, station_preds, _G, alpha=alpha_input)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "map_html": m._repr_html_(),
            "routes": [
                {"label": "Fastest",  "steps": fastest_steps, "dist_km": fastest_dist, "time_min": fastest_time, "pm25": fastest_mean_pm25},
                {"label": "Optimal",  "steps": optimal_steps, "dist_km": optimal_dist, "time_min": optimal_time, "pm25": optimal_mean_pm25},
            ],
            "start": start,
            "end": end,
            "used_alpha": alpha_input,
        })
    except Exception as e:
        m = generate_base_map()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "map_html": m._repr_html_(),
                "error": f"Could not compute route: {e}",
                "start": start,
                "end": end,
            },
        )
