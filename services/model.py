# Predicts PM2.5 30 minutes ahead (2 steps at 15-min intervals)
# using lag features + meteorological context + station identity

import glob
import os

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# config
DATA_ROOT = "."
MODEL_PATH = "pm25_model.joblib"
N_LAGS = 4          # 4 × 15 min = 1 hour of history
HORIZON = 2         # predict 2 steps ahead = 30 minutes
TARGET = "pm25"

FEATURES = [
    "pm25",
    "pm10",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "temperature",
    "relativehumidity",
]


# data loading
def load_all_data(data_root: str) -> pd.DataFrame:
    """Glob all CSVs, pivot long -> wide, return unified DataFrame."""
    pattern = os.path.join(data_root, "data-*-*", "month=*", "*.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(f"No CSVs found under {data_root!r}. Check DATA_ROOT.")

    print(f"Found {len(files)} CSV files across all stations/years.")
    chunks = []
    for f in files:
        try:
            chunks.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Skipping {f}: {e}")

    raw = pd.concat(chunks, ignore_index=True)
    print(f"Total rows before pivot: {len(raw):,}")

    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)

    df = raw.pivot_table(
        index=["datetime", "location_id", "lat", "lon"],
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()

    df.columns.name = None
    return df


# feature engineering
def engineer_features(df: pd.DataFrame, label_encoder: LabelEncoder | None = None):
    df = df.sort_values(["location_id", "datetime"]).copy()

    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["station_encoded"] = label_encoder.fit_transform(
            df["location_id"].astype(str)
        )
    else:
        df["station_encoded"] = label_encoder.transform(df["location_id"].astype(str))

    wind_rad = np.deg2rad(df["wind_direction"].fillna(0))
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)

    hour = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["datetime"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["datetime"].dt.dayofweek / 7)

    fill_cols = FEATURES + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    existing = [c for c in fill_cols if c in df.columns]
    df[existing] = df.groupby("location_id")[existing].transform(
        lambda s: s.ffill().bfill()
    )

    lag_cols = []
    for col in FEATURES:
        if col not in df.columns:
            continue
        for lag in range(1, N_LAGS + 1):
            name = f"{col}_lag{lag}"
            df[name] = df.groupby("location_id")[col].shift(lag)
            lag_cols.append(name)

    df["target"] = df.groupby("location_id")[TARGET].shift(-HORIZON)
    df = df.dropna(subset=lag_cols + ["target"])

    return df, lag_cols, label_encoder


# train
def train(data_root: str = DATA_ROOT):
    df = load_all_data(data_root)
    df, lag_cols, label_encoder = engineer_features(df)

    time_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    weather_cols = [
        c
        for c in [
            "wind_dir_sin",
            "wind_dir_cos",
            "wind_speed",
            "temperature",
            "relativehumidity",
        ]
        if c in df.columns
    ]
    feature_cols = lag_cols + time_cols + weather_cols + ["station_encoded"]

    station_coords = df.groupby("location_id")[["lat", "lon"]].first().reset_index()

    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"\nTest MAE:  {mae:.2f} µg/m³")
    print(f"Test RMSE: {rmse:.2f} µg/m³")

    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
            "label_encoder": label_encoder,
            "station_coords": station_coords,
        },
        MODEL_PATH,
    )
    print(f"Model saved -> {DATA_ROOT}/{MODEL_PATH}")

    return model, feature_cols, label_encoder


# using the model
# helpers
def load_model():
    payload = joblib.load(MODEL_PATH)
    return (
        payload["model"],
        payload["features"],
        payload["label_encoder"],
        payload["station_coords"],
    )


def _build_feature_row(
    rows: pd.DataFrame,
    station_id: int,
    label_encoder: LabelEncoder,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build a single feature row from the last N_LAGS readings in `rows`."""
    latest = rows.iloc[-1]
    row: dict = {}

    for col in FEATURES:
        if col not in rows.columns:
            continue
        for lag in range(1, N_LAGS + 1):
            row[f"{col}_lag{lag}"] = (
                rows.iloc[-lag][col] if col in rows.columns else 0.0
            )

    for col in [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "wind_dir_sin",
        "wind_dir_cos",
        "wind_speed",
        "temperature",
        "relativehumidity",
    ]:
        row[col] = latest.get(col, 0.0)

    row["station_encoded"] = label_encoder.transform([str(station_id)])[0]
    return pd.DataFrame([row])[feature_cols]


def _prepare_rows(recent_readings: list[dict]) -> pd.DataFrame:
    """Parse and enrich a list of raw reading dicts into a DataFrame."""
    rows = pd.DataFrame(recent_readings)
    rows["datetime"] = pd.to_datetime(rows["datetime"], utc=True)
    rows = rows.sort_values("datetime").reset_index(drop=True)

    wind_rad = np.deg2rad(rows["wind_direction"].fillna(0))
    rows["wind_dir_sin"] = np.sin(wind_rad)
    rows["wind_dir_cos"] = np.cos(wind_rad)

    hour = rows["datetime"].dt.hour + rows["datetime"].dt.minute / 60
    rows["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    rows["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    rows["dow_sin"] = np.sin(2 * np.pi * rows["datetime"].dt.dayofweek / 7)
    rows["dow_cos"] = np.cos(2 * np.pi * rows["datetime"].dt.dayofweek / 7)

    return rows


# single step inference
def predict_pm25(station_id: int, recent_readings: list[dict]) -> float:
    """
    Predict PM2.5 30 minutes from now for a single station.

    recent_readings: list of dicts (oldest -> newest), each with keys
        matching FEATURES plus 'datetime' and 'wind_direction'.
        Must have at least N_LAGS entries.
    """
    if len(recent_readings) < N_LAGS:
        raise ValueError(
            f"Need at least {N_LAGS} readings, got {len(recent_readings)}."
        )

    model, feature_cols, label_encoder, _ = load_model()
    rows = _prepare_rows(recent_readings[-N_LAGS:])
    X = _build_feature_row(rows, station_id, label_encoder, feature_cols)
    return float(model.predict(X)[0])


# # multi step inference
# def predict_pm25_sequence(
#     station_id: int,
#     recent_readings: list[dict],
#     n_steps: int = 10,
# ) -> list[dict]:
#     """
#     Autoregressively predict PM2.5 for the next n_steps x 15 minutes.

#     Returns a list of dicts, each with:
#         - 'step':     step index (1 = +15 min, 2 = +30 min, ...)
#         - 'minutes':  minutes from now
#         - 'pm25':     predicted PM2.5 (µg/m³)
#         - 'datetime': predicted UTC datetime

#     How it works:
#         The model was trained to predict HORIZON steps ahead (30 min).
#         For multi-step forecasting we use an autoregressive strategy:
#         each prediction is fed back as a new "reading" so the next
#         prediction can use it as a lag feature. Uncertainty compounds
#         with each step.
#     """
#     if len(recent_readings) < N_LAGS:
#         raise ValueError(
#             f"Need at least {N_LAGS} readings, got {len(recent_readings)}."
#         )

#     model, feature_cols, label_encoder, _ = load_model()

#     # Start with a rolling window of the most recent N_LAGS readings
#     window = list(recent_readings[-N_LAGS:])
#     rows = _prepare_rows(window)
#     # print(rows)
#     last_dt = rows["datetime"].iloc[-1]

#     results = []
#     for step in range(1, n_steps + 1):
#         X = _build_feature_row(rows, station_id, label_encoder, feature_cols)
#         # print(X)
#         predicted = float(model.predict(X)[0])
#         next_dt = last_dt + pd.Timedelta(minutes=15 * step)

#         results.append(
#             {
#                 "step": step,
#                 "minutes": 15 * step,
#                 "pm25": round(predicted, 2),
#                 "datetime": next_dt.isoformat(),
#             }
#         )

#         # Feed prediction back into the window as a synthetic reading,
#         # preserving the most recent meteorological context
#         latest = window[-1].copy()
#         latest["datetime"] = next_dt.isoformat()
#         latest["pm25"] = predicted
#         # pm10 roughly tracks pm25 -- use last known ratio to approximate
#         if window[-1].get("pm10") and window[-1].get("pm25"):
#             ratio = window[-1]["pm10"] / max(window[-1]["pm25"], 1)
#             latest["pm10"] = predicted * ratio
#         window.append(latest)
#         rows = _prepare_rows(window[-N_LAGS:])

#     return results


# spatial inference
def interpolate_pm25_for_nodes(
    node_coords: list[tuple[int, float, float]],
    station_predictions: dict[int, float],
) -> dict[int, float]:
    """
    Estimate PM2.5 at arbitrary road graph nodes via inverse distance
    weighting (IDW) from known station predictions.

    node_coords:         [(node_id, lat, lon), ...]
    station_predictions: {station_id: predicted_pm25}

    Returns {node_id: estimated_pm25}.
    """
    _, _, _, station_coords_df = load_model()

    station_lats, station_lons, station_vals = [], [], []
    for _, row in station_coords_df.iterrows():
        sid = int(row["location_id"])
        if sid in station_predictions:
            station_lats.append(row["lat"])
            station_lons.append(row["lon"])
            station_vals.append(station_predictions[sid])

    if not station_vals:
        raise ValueError("No station predictions matched stored station coordinates.")

    s_lats = np.array(station_lats)
    s_lons = np.array(station_lons)
    s_vals = np.array(station_vals)

    node_pm25 = {}
    for node_id, n_lat, n_lon in node_coords:
        dists = np.sqrt((s_lats - n_lat) ** 2 + (s_lons - n_lon) ** 2)
        if dists.min() < 1e-6:
            node_pm25[node_id] = float(s_vals[dists.argmin()])
            continue
        weights = 1.0 / (dists**2)
        node_pm25[node_id] = float(np.sum(weights * s_vals) / np.sum(weights))

    return node_pm25


if __name__ == "__main__":
    train()
