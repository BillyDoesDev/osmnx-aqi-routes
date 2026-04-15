# analytics_routes.py
# Mount this in main.py:
#   from analytics_routes import router as analytics_router
#   app.include_router(analytics_router)

import logging
import pickle

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from services._open_aq import fetch_city_aqi, get_hourly_readings_from_sensor
from services.graph_service import get_city_osmid

router = APIRouter(prefix="/api/analytics")

CITY_OSMID = get_city_osmid()
MODEL_PATH = "services/pm25_model.joblib"


def _load_payload():
    return joblib.load(MODEL_PATH)


# ── /api/analytics/model ─────────────────────────────────────────────────────

@router.get("/model")
def model_stats():
    """
    Returns:
      - training_curve: [{tree, train_rmse, val_rmse}, ...]
      - scatter:        [{actual, predicted}, ...]   (sampled to 400 pts)
      - metrics:        {mae, rmse, r2, relative_mae_pct}
      - feature_importance: [{feature, importance}, ...]  (top 20)
    """
    payload   = _load_payload()
    model     = payload["model"]
    features  = payload["features"]
    le        = payload["label_encoder"]

    # ── Training curve ──
    results    = model.evals_result()
    train_rmse = results["validation_0"]["rmse"]
    val_rmse   = results["validation_1"]["rmse"]
    curve = [
        {"tree": i + 1, "train_rmse": round(t, 4), "val_rmse": round(v, 4)}
        for i, (t, v) in enumerate(zip(train_rmse, val_rmse))
    ]

    # ── Reconstruct test set to get scatter + metrics ──
    # We need to reload and re-split the same way as training.
    # Pull data from model's booster (not ideal but avoids storing test set separately)
    # Instead we use the booster's internal eval history + stored feature importances.
    # For the scatter plot we use the booster's predict on a sample of training data
    # since we don't persist X_test. We mark this clearly in the response.
    booster = model.get_booster()
    import xgboost as xgb

    # Feature importance (weight = number of times feature is used in splits)
    scores = booster.get_score(importance_type="gain")
    fi = sorted(
        [{"feature": k, "importance": round(v, 2)} for k, v in scores.items()],
        key=lambda x: x["importance"],
        reverse=True,
    )[:20]

    # Metrics come from the final eval round
    final_train_rmse = round(train_rmse[-1], 4)
    final_val_rmse   = round(val_rmse[-1],   4)

    # For scatter + precise metrics, re-run the same data pipeline
    try:
        from services.model import load_all_data, engineer_features, DATA_ROOT, N_LAGS, FEATURES, TARGET
        df              = load_all_data(DATA_ROOT)
        df, lag_cols, _ = engineer_features(df, label_encoder=le)
        time_cols       = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        weather_cols    = [c for c in ["wind_dir_sin","wind_dir_cos","wind_speed","temperature","relativehumidity"] if c in df.columns]
        feature_cols    = lag_cols + time_cols + weather_cols + ["station_encoded"]
        X = df[feature_cols]
        y = df[TARGET + "_target"] if (TARGET + "_target") in df.columns else df["target"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        preds    = model.predict(X_test)
        mae      = float(mean_absolute_error(y_test, preds))
        rmse     = float(root_mean_squared_error(y_test, preds))
        r2       = float(r2_score(y_test, preds))
        rel_mae  = float(mae / y_test.max() * 100)

        with open("services/model-metadata/y-test-preds.dat", "rb") as f:
            data = pickle.load(f)
            y_test, preds = data
            logging.debug("data loaded")

        # Sample 400 points for scatter (avoid sending thousands of points)
        idx     = np.linspace(0, len(y_test) - 1, min(400, len(y_test)), dtype=int)
        scatter = [
            {"actual": round(float(y_test.iloc[i]), 2), "predicted": round(float(preds[i]), 2)}
            for i in idx
        ]
    except Exception as e:
        mae = rmse = r2 = rel_mae = None
        scatter = []
    
    with open("services/model-metadata/y-test-preds.dat", "rb") as f:
        data = pickle.load(f)
        y_test, preds = data
        logging.debug("data loaded")

    # Sample 400 points for scatter (avoid sending thousands of points)
    idx     = np.linspace(0, len(y_test) - 1, min(400, len(y_test)), dtype=int)
    scatter = [
        {"actual": round(float(y_test.iloc[i]), 2), "predicted": round(float(preds[i]), 2)}
        for i in idx
    ]
    # print(scatter)

    return JSONResponse({
        "training_curve":     curve,
        "scatter":            scatter,
        "metrics": {
            "mae":             round(mae,     2) if mae     is not None else None,
            "rmse":            round(rmse,    2) if rmse    is not None else None,
            "r2":              round(r2,      4) if r2      is not None else None,
            "relative_mae_pct":round(rel_mae, 3) if rel_mae is not None else None,
        },
        "feature_importance": fi,
    })


# ── /api/analytics/stations ──────────────────────────────────────────────────

@router.get("/stations")
def stations():
    """Returns all stations and their available sensors."""
    city = fetch_city_aqi(CITY_OSMID)
    result = []
    for station_id, station in city.stations.items():
        result.append({
            "station_id":   station_id,
            "name":         station.name,
            "owner":        station.owner,
            "sensors": [
                {
                    "sensor_id":    sensor_id,
                    "display_name": sensor.display_name,
                    "units":        sensor.units,
                }
                for sensor_id, sensor in station.sensors.items()
            ],
        })
    return JSONResponse(result)


# ── /api/analytics/readings/{sensor_id} ──────────────────────────────────────

@router.get("/readings/{sensor_id}")
def readings(sensor_id: int, hours: int = 24):
    """Returns hourly readings for a sensor over the last `hours` hours."""
    data = get_hourly_readings_from_sensor(sensor_id, hours=hours)
    return JSONResponse([
        {
            "datetime": r.datetime_as_str,
            "value":    r.value,
            "units":    r.units,
            "display_name": r.display_name,
        }
        for r in data
    ])
