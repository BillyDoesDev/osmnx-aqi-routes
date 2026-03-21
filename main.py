from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from services.graph_service import generate_base_map

app = FastAPI()
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
