To get started with the right dependencies, cd into this dir and then
```bash
uv sync
```

To run, do:
```bash
uvicorn main:app --reload
```

Services used so far:

- fastapi: backend
- osmnx: main routing engine + graph
- folium: map rendering
