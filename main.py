from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from services.graph_service import generate_base_map, get_boundary
import json
import logging


logging.basicConfig(
    format="[{levelname}] {asctime}: {message}", style="{", level=logging.DEBUG
)

city = None
_boundary_geojson = None
_boundary_shape = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.debug("Setting up lifespan...")
    global city, _boundary_geojson, _boundary_shape
    city, _boundary_geojson, _boundary_shape = get_boundary()
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
