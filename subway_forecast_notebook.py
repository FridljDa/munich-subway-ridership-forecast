# %% [markdown]
# # Munich Subway Ridership Forecast
#
# This notebook-style script builds a leakage-safe, explainable forecast for hourly station
# passenger counts, exactly 7 days ahead.
#
# The audience is non-technical leadership: the model is intentionally a strong tabular
# baseline rather than a complex deep-learning architecture. LightGBM gives good accuracy on mixed
# tabular features, handles categorical variables natively, trains quickly, and has direct
# interpretability via feature importance and partial dependence.

# %%

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# %%

RANDOM_SEED = 42
FORECAST_HORIZON_HOURS = 168
OPERATING_HOURS = tuple(range(7, 13))
LOCAL_TZ = "Europe/Berlin"
STATION_SNAPSHOT_PREDICTED_FOR_LOCAL = pd.Timestamp("2026-02-02 11:00:00", tz=LOCAL_TZ)

CATEGORICAL_FEATURES = [
    "station_id",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_octoberfest",
    "is_interchange",
]

NUMERIC_FEATURES = [
    "station_capacity",
    "connectivity_score",
    "traffic_index_closest_lag_168h",
    "passenger_count_lag_168h",
    "passenger_count_trailing_4_weeks_same_hour_average",
    "temp_C_prediction",
    "precip_mm_prediction",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "passenger_count"
NAIVE_BASELINE_FEATURE = "passenger_count_trailing_4_weeks_same_hour_average"

SLIDE_COLORS = {
    "accent": "#0f7a64",
    "accent_light": "#5fb89f",
    "accent_pale": "#cfe3db",
    "contrast": "#b4543b",
    "text": "#1b1a17",
    "muted": "#57554e",
    "muted_light": "#8a8578",
    "grid": "#e4e0d7",
}

SERIES_COLORS = [
    SLIDE_COLORS["accent"],
    SLIDE_COLORS["contrast"],
    SLIDE_COLORS["muted"],
    SLIDE_COLORS["accent_pale"],
    SLIDE_COLORS["muted_light"],
]

matplotlib.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.edgecolor": "none",
        "text.color": SLIDE_COLORS["text"],
        "axes.labelcolor": SLIDE_COLORS["text"],
        "axes.edgecolor": SLIDE_COLORS["grid"],
        "axes.titlecolor": SLIDE_COLORS["text"],
        "xtick.color": SLIDE_COLORS["muted"],
        "ytick.color": SLIDE_COLORS["muted"],
        "grid.color": SLIDE_COLORS["grid"],
        "grid.alpha": 0.8,
        "axes.prop_cycle": plt.cycler(color=SERIES_COLORS),
    }
)


def notebook_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent

    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    raise RuntimeError("Cannot locate data/ from the current working directory.")


BASE_DIR = notebook_dir()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

LEGACY_ARTIFACTS = [
    OUTPUT_DIR / "rmse_by_station_with_smart2.csv",
    FIGURE_DIR / "rmse_by_station_with_smart2.png",
]

for path in LEGACY_ARTIFACTS:
    if path.exists():
        path.unlink()


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLIDE_COLORS["grid"])
    ax.spines["bottom"].set_color(SLIDE_COLORS["grid"])


def style_legend(ax: plt.Axes, **kwargs: object) -> None:
    legend = ax.legend(frameon=False, **kwargs)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(SLIDE_COLORS["muted"])


def style_figure_title(ax: plt.Axes, title: str, subtitle: str) -> float:
    """Place title and subtitle above the axes; return top for tight_layout rect."""
    fig = ax.figure
    subtitle_lines = subtitle.count("\n") + 1
    headroom = 0.052 + 0.026 * subtitle_lines

    fig.text(
        0.0,
        1.0,
        title,
        transform=fig.transFigure,
        fontsize=14,
        color=SLIDE_COLORS["text"],
        va="top",
        ha="left",
    )
    fig.text(
        0.0,
        1.0 - 0.040,
        subtitle,
        transform=fig.transFigure,
        fontsize=10,
        color=SLIDE_COLORS["muted"],
        va="top",
        ha="left",
        linespacing=1.35,
    )
    return 1.0 - headroom


def save_figure(fig: plt.Figure, path: Path, **kwargs: object) -> None:
    fig.savefig(path, dpi=180, transparent=True, **kwargs)


def save_figure_csv(name: str, frame: pd.Series | pd.DataFrame) -> None:
    output = frame.to_frame() if isinstance(frame, pd.Series) else frame.copy()
    output.to_csv(OUTPUT_DIR / f"{name}.csv", index=True)


def add_prediction_origin_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    predicted_for_local = pd.to_datetime(result["predicted_for_local"])
    result["prediction_origin_timestamp"] = predicted_for_local - pd.Timedelta(
        hours=FORECAST_HORIZON_HOURS
    )
    return result


PREDICTION_OUTPUT_COLUMNS = [
    "station_id",
    "station_name",
    "station_display_name",
    "predicted_for_local",
    "predicted_for_utc",
    "prediction_origin_timestamp",
    "actual_passenger_count",
    "predicted_passenger_count",
    "baseline_predicted_passenger_count",
    "absolute_error",
]

# %% [markdown]
# ## Calendar Features
#
# In production these would be maintained as small reference tables in the data lake and refreshed
# annually. For this synthetic dataset, deterministic local definitions keep the notebook
# self-contained and reproducible.

# %%

OKTOBERFEST_PERIODS = (
    ("2023-09-16", "2023-10-03"),
    ("2024-09-21", "2024-10-06"),
    ("2025-09-20", "2025-10-05"),
    ("2026-09-19", "2026-10-04"),
)


def easter_sunday(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday via the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    correction = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * correction) // 451
    month = (h + correction - 7 * m + 114) // 31
    day = ((h + correction - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def bavaria_holidays(years: range) -> set[pd.Timestamp]:
    holidays: set[pd.Timestamp] = set()
    for year in years:
        easter = easter_sunday(year)
        holidays.update(
            {
                pd.Timestamp(year=year, month=1, day=1),  # New Year's Day
                pd.Timestamp(year=year, month=1, day=6),  # Epiphany
                easter - pd.Timedelta(days=2),  # Good Friday
                easter + pd.Timedelta(days=1),  # Easter Monday
                pd.Timestamp(year=year, month=5, day=1),  # Labour Day
                easter + pd.Timedelta(days=39),  # Ascension Day
                easter + pd.Timedelta(days=50),  # Whit Monday
                easter + pd.Timedelta(days=60),  # Corpus Christi
                pd.Timestamp(year=year, month=8, day=15),  # Assumption Day
                pd.Timestamp(year=year, month=10, day=3),  # German Unity Day
                pd.Timestamp(year=year, month=11, day=1),  # All Saints' Day
                pd.Timestamp(year=year, month=12, day=25),  # Christmas Day
                pd.Timestamp(year=year, month=12, day=26),  # Boxing Day
            }
        )
    return holidays


HOLIDAY_DATES = bavaria_holidays(range(2023, 2027))


def is_octoberfest_date(date: pd.Timestamp) -> bool:
    date = pd.Timestamp(date).normalize()
    return any(
        pd.Timestamp(start) <= date <= pd.Timestamp(end) for start, end in OKTOBERFEST_PERIODS
    )


# %% [markdown]
# ## Load Data

# %%


def read_csv_with_timestamp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def format_station_display_name(name: str) -> str:
    return name.replace("_", " ")


station_metadata = pd.read_csv(DATA_DIR / "station_metadata.csv")
station_metadata["station_display_name"] = station_metadata["station_name"].map(
    format_station_display_name
)
timeseries = read_csv_with_timestamp(DATA_DIR / "timeseries_with_target.csv")
weather = read_csv_with_timestamp(DATA_DIR / "weather_hourly.csv")
traffic = read_csv_with_timestamp(DATA_DIR / "traffic_hourly.csv")

print("Loaded data")
print(f"stations: {station_metadata.shape}")
print(f"target rows: {timeseries.shape}")
print(f"weather rows: {weather.shape}")
print(f"traffic rows: {traffic.shape}")


# %% [markdown]
# ## Feature Engineering
#
# Every row represents one target timestamp. The 168-hour lag and trailing weekly features are built
# relative to that timestamp, so the test labels themselves are never used as predictors.

# %%


def add_local_time_columns(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    result = df.copy()
    local_ts = result[timestamp_col].dt.tz_convert(LOCAL_TZ)
    result["local_timestamp"] = local_ts
    result["date"] = local_ts.dt.normalize().dt.tz_localize(None)
    result["hour"] = local_ts.dt.hour
    result["day_of_week"] = local_ts.dt.dayofweek
    result["month"] = local_ts.dt.month
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype("int8")
    result["is_holiday"] = result["date"].isin(HOLIDAY_DATES).astype("int8")
    result["is_octoberfest"] = result["date"].map(is_octoberfest_date).astype("int8")
    return result


def nearest_traffic_sensor_map(
    stations: pd.DataFrame,
    traffic_data: pd.DataFrame,
) -> pd.DataFrame:
    sensors = traffic_data[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    rows = []
    for station in stations.itertuples(index=False):
        distances = (sensors["lat"] - station.lat) ** 2 + (sensors["lon"] - station.lon) ** 2
        nearest = sensors.loc[distances.idxmin()]
        rows.append(
            {
                "station_id": station.station_id,
                "traffic_lat": nearest["lat"],
                "traffic_lon": nearest["lon"],
            }
        )
    return pd.DataFrame(rows)


def add_passenger_lags(base: pd.DataFrame) -> pd.DataFrame:
    result = base.copy()
    historical = base[["station_id", "timestamp", TARGET]].copy()
    for weeks in range(1, 5):
        lag_hours = FORECAST_HORIZON_HOURS * weeks
        lag_col = f"passenger_count_lag_{lag_hours}h"
        lagged = historical.rename(
            columns={
                "timestamp": f"timestamp_lag_{lag_hours}h",
                TARGET: lag_col,
            }
        )
        result[f"timestamp_lag_{lag_hours}h"] = result["timestamp"] - pd.Timedelta(hours=lag_hours)
        result = result.merge(
            lagged,
            on=["station_id", f"timestamp_lag_{lag_hours}h"],
            how="left",
        )

    weekly_lag_cols = [
        f"passenger_count_lag_{FORECAST_HORIZON_HOURS * weeks}h" for weeks in range(1, 5)
    ]
    result["passenger_count_trailing_4_weeks_same_hour_average"] = result[weekly_lag_cols].mean(
        axis=1,
        skipna=True,
    )
    return result.rename(columns={"passenger_count_lag_168h": "passenger_count_lag_168h"})


def build_feature_frame(
    ts: pd.DataFrame,
    stations: pd.DataFrame,
    weather_data: pd.DataFrame,
    traffic_data: pd.DataFrame,
) -> pd.DataFrame:
    features = add_local_time_columns(ts)
    features = features.merge(
        stations[
            [
                "station_id",
                "station_name",
                "station_display_name",
                "lat",
                "lon",
                "station_capacity",
                "is_interchange",
                "connectivity_score",
            ]
        ],
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    weather_features = weather_data.rename(
        columns={"temp_C": "temp_C_prediction", "precip_mm": "precip_mm_prediction"}
    )[["timestamp", "temp_C_prediction", "precip_mm_prediction"]]
    features = features.merge(weather_features, on="timestamp", how="left", validate="many_to_one")

    traffic_sensor_map = nearest_traffic_sensor_map(stations, traffic_data)
    traffic_lag = traffic_data.rename(
        columns={
            "timestamp": "traffic_timestamp_lag_168h",
            "lat": "traffic_lat",
            "lon": "traffic_lon",
            "traffic_index": "traffic_index_closest_lag_168h",
        }
    )
    features = features.merge(
        traffic_sensor_map, on="station_id", how="left", validate="many_to_one"
    )
    features["traffic_timestamp_lag_168h"] = features["timestamp"] - pd.Timedelta(
        hours=FORECAST_HORIZON_HOURS
    )
    features = features.merge(
        traffic_lag,
        on=["traffic_timestamp_lag_168h", "traffic_lat", "traffic_lon"],
        how="left",
        validate="many_to_one",
    )

    features = add_passenger_lags(features)
    return features


feature_frame = build_feature_frame(timeseries, station_metadata, weather, traffic)

last_timestamp = feature_frame["timestamp"].max()

# Three-way split that mirrors the production Job cycle:
#
#  train_frame      [2023-01-01 → retrain_cutoff]   Job 3 trains here; this is what the
#                                                    deployed model has ever seen.
#  inference window (retrain_cutoff, test_start)     Job 1 runs here. The lag feature values
#                                                    for test rows come from this window (e.g.
#                                                    lag_168h for a Jan 27 target = the Jan 20
#                                                    count, observed at T_run = Jan 20). The
#                                                    model itself is never trained on these rows.
#  test_frame       [test_start → last_timestamp]   The predictions we evaluate; targets that
#                                                    Job 1 forecasts during the inference window.
#
# retrain_cutoff = 14 days before the last data point:
#   last_timestamp (local) = 2026-02-02 12:00  →  retrain_cutoff = 2026-01-19 12:00
# test_start = 7 days before the last data point (+1 h to skip the non-operating gap):
#   test_start = 2026-01-26 13:00 (first test row: 2026-01-27 07:00)
retrain_cutoff = last_timestamp - pd.Timedelta(days=14)
test_start = last_timestamp - pd.Timedelta(days=7) + pd.Timedelta(hours=1)
train_mask = feature_frame["timestamp"] <= retrain_cutoff
test_mask = feature_frame["timestamp"] >= test_start

train_frame = feature_frame.loc[train_mask].copy()
test_frame = feature_frame.loc[test_mask].copy()

print(f"Train rows:  {len(train_frame):,}")
print(
    f"Train window: {train_frame['local_timestamp'].min()} to {train_frame['local_timestamp'].max()}"
)
print(f"Test rows:   {len(test_frame):,}")
print(
    f"Test window:  {test_frame['local_timestamp'].min()} to {test_frame['local_timestamp'].max()}"
)


# %% [markdown]
# ## Model Utilities

# %%


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    # RMSE = sqrt(mean((y_true - y_pred)^2))
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_array - y_pred_array) ** 2)))


def station_snapshot_subtitle(train_cutoff_local: pd.Timestamp) -> str:
    predicted_at = STATION_SNAPSHOT_PREDICTED_FOR_LOCAL - pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    return (
        f"Predicted at {predicted_at:%Y-%m-%d %H:%M} "
        f"for {STATION_SNAPSHOT_PREDICTED_FOR_LOCAL:%Y-%m-%d %H:%M} (local). "
        f"Model trained through {train_cutoff_local:%Y-%m-%d %H:%M}."
    )


def format_observation_count(n_values: pd.Series) -> str:
    n_min = int(n_values.min())
    n_max = int(n_values.max())
    if n_min == n_max:
        return f"{n_min} hourly observations"
    return f"{n_min}-{n_max} hourly observations"


def station_rmse_figure_subtitle(
    last_predicted_week_start: pd.Timestamp,
    last_predicted_week_end: pd.Timestamp,
    n_values: pd.Series,
    train_cutoff_local: pd.Timestamp,
) -> str:
    week_range = f"{last_predicted_week_start:%Y-%m-%d} to {last_predicted_week_end:%Y-%m-%d}"
    return (
        f"Weekly average for last holdout week ({week_range}, "
        f"{format_observation_count(n_values)}). "
        f"Model trained through {train_cutoff_local:%Y-%m-%d %H:%M}."
    )


def compute_last_week_station_rmse_results(test_output: pd.DataFrame) -> pd.DataFrame:
    evaluated = test_output[test_output["actual_passenger_count"].notna()].copy()
    predicted_for_local = pd.to_datetime(evaluated["predicted_for_local"])
    predicted_for_dates = predicted_for_local.dt.normalize()
    last_predicted_week_end = predicted_for_dates.max()
    last_predicted_week_start = last_predicted_week_end - pd.Timedelta(days=6)
    last_week_output = evaluated[
        (predicted_for_dates >= last_predicted_week_start)
        & (predicted_for_dates <= last_predicted_week_end)
    ]

    # Per-station RMSE on the last predicted week, using weekly mean actual vs weekly mean forecast:
    #   avg_actual_s = mean(actual hourly counts in that week for station s)
    #   avg_pred_s   = mean(predicted hourly counts in that week for station s)
    #   RMSE_s = sqrt(mean((avg_actual_s - avg_pred_s)^2)) = |avg_actual_s - avg_pred_s|
    station_results = []
    for station_id, station_test in last_week_output.groupby("station_id", observed=True):
        avg_actual = station_test["actual_passenger_count"].mean()
        station_results.append(
            {
                "station_id": station_id,
                "station_name": station_test["station_name"].iloc[0],
                "station_display_name": station_test["station_display_name"].iloc[0],
                "last_predicted_week_start": last_predicted_week_start,
                "last_predicted_week_end": last_predicted_week_end,
                "rmse": rmse([avg_actual], [station_test["predicted_passenger_count"].mean()]),
                "baseline_rmse": rmse(
                    [avg_actual],
                    [station_test["baseline_predicted_passenger_count"].mean()],
                ),
                "n": len(station_test),
                "missing_actuals": 0,
            }
        )
    return pd.DataFrame(station_results).sort_values("rmse")


def add_forecast_noise(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    for column in ["temp_C_prediction", "precip_mm_prediction"]:
        sigma = float(reference[column].std(ddof=0) / 5)
        if sigma > 0:
            result[column] = result[column] + rng.normal(0, sigma, size=len(result))
    result["precip_mm_prediction"] = result["precip_mm_prediction"].clip(lower=0)
    return result


def fill_missing_features(
    train: pd.DataFrame,
    other: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, float]]:
    train_result = train.copy()
    other_result = None if other is None else other.copy()
    fill_values: dict[str, float] = {}

    for column in NUMERIC_FEATURES:
        value = float(train_result[column].median())
        fill_values[column] = value
        train_result[column] = train_result[column].fillna(value)
        if other_result is not None:
            other_result[column] = other_result[column].fillna(value)

    return train_result, other_result, fill_values


def as_model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame[FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        matrix[column] = matrix[column].astype("category")
    return matrix


def fit_baseline(train: pd.DataFrame, predict_frame: pd.DataFrame) -> np.ndarray:
    train = train.dropna(subset=[TARGET]).copy()
    global_mean = float(train[TARGET].mean())
    predictions = predict_frame[NAIVE_BASELINE_FEATURE].astype(float).fillna(global_mean).to_numpy()
    return np.clip(predictions, 0, None)


def lightgbm_params() -> dict[str, object]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 60,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "feature_pre_filter": False,
    }


@dataclass(frozen=True)
class TrainedModel:
    booster: lgb.Booster
    fill_values: dict[str, float]


def train_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame | None = None,
) -> TrainedModel:
    train = train.dropna(subset=[TARGET]).copy()
    if valid is not None:
        valid = valid.dropna(subset=[TARGET]).copy()

    noisy_train = add_forecast_noise(train, train)
    filled_train, filled_valid, fill_values = fill_missing_features(noisy_train, valid)

    train_dataset = lgb.Dataset(
        as_model_matrix(filled_train),
        label=filled_train[TARGET],
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    valid_sets = [train_dataset]
    valid_names = ["train"]
    callbacks: list[object] = [lgb.log_evaluation(period=0)]

    if filled_valid is not None:
        valid_dataset = lgb.Dataset(
            as_model_matrix(filled_valid),
            label=filled_valid[TARGET],
            categorical_feature=CATEGORICAL_FEATURES,
            reference=train_dataset,
            free_raw_data=False,
        )
        valid_sets.append(valid_dataset)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

    booster = lgb.train(
        lightgbm_params(),
        train_dataset,
        num_boost_round=600,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return TrainedModel(booster=booster, fill_values=fill_values)


def predict_lightgbm(model: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    matrix_frame = frame.copy()
    for column, value in model.fill_values.items():
        matrix_frame[column] = matrix_frame[column].fillna(value)
    predictions = model.booster.predict(
        as_model_matrix(matrix_frame),
        num_iteration=model.booster.best_iteration or model.booster.current_iteration(),
    )
    return np.clip(predictions, 0, None)


# %% [markdown]
# ## Exploratory Figures

# %%


def save_eda_figures(
    frame: pd.DataFrame,
    raw_timeseries: pd.DataFrame,
    stations: pd.DataFrame,
) -> None:
    hourly = frame.groupby("hour", observed=True)[TARGET].mean()
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_average = frame.groupby("day_of_week", observed=True)[TARGET].mean().reindex(range(7))
    weekday_average.index = weekday_labels
    station_average = (
        frame.groupby("station_display_name", observed=True)[TARGET]
        .mean()
        .sort_values(ascending=False)
    )
    station_hour = (
        frame.groupby(["station_display_name", "hour"], observed=True)[TARGET]
        .mean()
        .unstack("hour")
    )
    hourly_df = hourly.rename("average_passenger_count").rename_axis("hour")
    weekday_average_df = weekday_average.rename("average_passenger_count").rename_axis("weekday")
    station_average_df = station_average.rename("average_passenger_count").rename_axis(
        "station_display_name"
    )
    station_hour_df = station_hour.rename_axis(index="station_display_name", columns="hour")

    fig_hourly, ax_hourly = plt.subplots(figsize=(7, 5))
    hourly.plot(kind="bar", ax=ax_hourly, color=SLIDE_COLORS["accent"], edgecolor="none")
    ax_hourly.set_title("Average ridership by operating hour")
    ax_hourly.set_xlabel("Local hour")
    ax_hourly.set_ylabel("Passengers")
    style_axis(ax_hourly)
    fig_hourly.tight_layout()
    save_figure(fig_hourly, FIGURE_DIR / "eda_average_ridership_by_hour.png")
    save_figure_csv("eda_average_ridership_by_hour", hourly_df)
    plt.close(fig_hourly)

    fig_weekday, ax_weekday = plt.subplots(figsize=(7, 5))
    weekday_average.plot(
        kind="bar", ax=ax_weekday, color=SLIDE_COLORS["contrast"], edgecolor="none"
    )
    ax_weekday.set_title("Average ridership by weekday")
    ax_weekday.set_xlabel("Weekday")
    ax_weekday.set_ylabel("Passengers")
    style_axis(ax_weekday)
    fig_weekday.tight_layout()
    save_figure(fig_weekday, FIGURE_DIR / "eda_average_ridership_by_weekday.png")
    save_figure_csv("eda_average_ridership_by_weekday", weekday_average_df)
    plt.close(fig_weekday)

    fig_station_average, ax_station_average = plt.subplots(figsize=(9, 5))
    station_average.plot(
        kind="bar", ax=ax_station_average, color=SLIDE_COLORS["accent_light"], edgecolor="none"
    )
    ax_station_average.set_title("Average ridership by station")
    ax_station_average.set_xlabel("Station")
    ax_station_average.set_ylabel("Passengers")
    ax_station_average.tick_params(axis="x", rotation=35)
    for label in ax_station_average.get_xticklabels():
        label.set_horizontalalignment("right")
    style_axis(ax_station_average)
    fig_station_average.tight_layout()
    save_figure(fig_station_average, FIGURE_DIR / "eda_average_ridership_by_station.png")
    save_figure_csv("eda_average_ridership_by_station", station_average_df)
    plt.close(fig_station_average)

    fig_station_profile, ax_station_profile = plt.subplots(figsize=(8, 5))
    station_hour.plot(ax=ax_station_profile, linewidth=1.8)
    ax_station_profile.set_title("Station-specific intraday profile")
    ax_station_profile.set_xlabel("Local hour")
    ax_station_profile.set_ylabel("Passengers")
    style_axis(ax_station_profile)
    style_legend(ax_station_profile, fontsize=7, ncols=2)
    fig_station_profile.tight_layout()
    save_figure(fig_station_profile, FIGURE_DIR / "eda_station_intraday_profile.png")
    save_figure_csv("eda_station_intraday_profile", station_hour_df)
    plt.close(fig_station_profile)

    odeonsplatz = (
        add_local_time_columns(raw_timeseries)
        .merge(
            stations[["station_id", "station_name"]],
            on="station_id",
            how="left",
            validate="many_to_one",
        )
        .loc[lambda df: df["station_name"] == "Odeonsplatz"]
        .sort_values("local_timestamp")
    )
    fig_odeonsplatz, ax_odeonsplatz = plt.subplots(figsize=(12, 5))
    ax_odeonsplatz.plot(
        odeonsplatz["local_timestamp"],
        odeonsplatz[TARGET],
        color=SLIDE_COLORS["accent"],
        linewidth=1.4,
    )
    ax_odeonsplatz.set_title("Odeonsplatz passenger count over time")
    ax_odeonsplatz.set_xlabel("Local timestamp")
    ax_odeonsplatz.set_ylabel("Passengers")
    style_axis(ax_odeonsplatz)
    fig_odeonsplatz.autofmt_xdate()
    fig_odeonsplatz.tight_layout()
    save_figure(fig_odeonsplatz, FIGURE_DIR / "eda_passenger_counts_odeonsplatz_overtime.png")
    save_figure_csv(
        "eda_passenger_counts_odeonsplatz_overtime",
        odeonsplatz[["local_timestamp", "timestamp", TARGET]].rename(
            columns={TARGET: "actual_passenger_count"}
        ),
    )
    plt.close(fig_odeonsplatz)


save_eda_figures(train_frame, timeseries, station_metadata)


# %% [markdown]
# ## Walk-Forward Validation
#
# Rolling one-week folds mimic how the model is used: train on the past, validate on the next week.

# %%


def walk_forward_validation(frame: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    rows = []
    last_train_timestamp = train_frame["timestamp"].max()
    for fold_index in range(n_folds, 0, -1):
        valid_end = last_train_timestamp - pd.Timedelta(days=7 * (fold_index - 1))
        valid_start = valid_end - pd.Timedelta(days=7) + pd.Timedelta(hours=1)

        fold_train = frame[frame["timestamp"] < valid_start].copy()
        fold_valid = frame[
            (frame["timestamp"] >= valid_start) & (frame["timestamp"] <= valid_end)
        ].copy()
        fold_valid_eval = fold_valid.dropna(subset=[TARGET]).copy()
        if fold_valid_eval.empty:
            continue

        baseline_pred = fit_baseline(fold_train, fold_valid_eval)
        model = train_lightgbm(fold_train, fold_valid_eval)
        model_pred = predict_lightgbm(model, fold_valid_eval)
        rows.append(
            {
                "fold": len(rows) + 1,
                "start_window": fold_train["local_timestamp"].min(),
                "end_window": fold_train["local_timestamp"].max(),
                "number_of_observations": len(fold_train.dropna(subset=[TARGET])),
                "valid_start": fold_valid["local_timestamp"].min(),
                "valid_end": fold_valid["local_timestamp"].max(),
                "n": len(fold_valid_eval),
                "missing_actuals": int(fold_valid[TARGET].isna().sum()),
                "rmse": rmse(fold_valid_eval[TARGET], model_pred),
                "baseline_rmse": rmse(fold_valid_eval[TARGET], baseline_pred),
                "best_iteration": model.booster.best_iteration or model.booster.current_iteration(),
            }
        )

    return pd.DataFrame(rows)


walk_forward_results = walk_forward_validation(feature_frame)
walk_forward_results.to_csv(OUTPUT_DIR / "walk_forward_validation.csv", index=False)
print(walk_forward_results)


# %% [markdown]
# ## Final Model and Test Evaluation

# %%

latest_validation_start = (
    train_frame["timestamp"].max() - pd.Timedelta(days=7) + pd.Timedelta(hours=1)
)
final_train_inner = train_frame[train_frame["timestamp"] < latest_validation_start].copy()
final_valid_inner = train_frame[train_frame["timestamp"] >= latest_validation_start].copy()

baseline_test_predictions = fit_baseline(train_frame, test_frame)
final_model = train_lightgbm(final_train_inner, final_valid_inner)
test_predictions = predict_lightgbm(final_model, test_frame)
test_eval_mask = test_frame[TARGET].notna()

test_output = test_frame[
    ["station_id", "station_name", "station_display_name", "local_timestamp", "timestamp", TARGET]
].copy()
test_output = test_output.rename(
    columns={
        "local_timestamp": "predicted_for_local",
        "timestamp": "predicted_for_utc",
        TARGET: "actual_passenger_count",
    }
)
test_output["predicted_passenger_count"] = test_predictions
test_output["baseline_predicted_passenger_count"] = baseline_test_predictions
test_output["absolute_error"] = (
    test_output["actual_passenger_count"] - test_output["predicted_passenger_count"]
).abs()
test_output = add_prediction_origin_timestamp(test_output)
test_output = test_output[PREDICTION_OUTPUT_COLUMNS]
test_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

overall_results = pd.DataFrame(
    [
        {
            "rmse": rmse(
                test_frame.loc[test_eval_mask, TARGET], test_predictions[test_eval_mask.to_numpy()]
            ),
            "baseline_rmse": rmse(
                test_frame.loc[test_eval_mask, TARGET],
                baseline_test_predictions[test_eval_mask.to_numpy()],
            ),
            "n": int(test_eval_mask.sum()),
            "missing_actuals": int((~test_eval_mask).sum()),
            "model": "LightGBM regression",
            "baseline": "passenger_count_trailing_4_weeks_same_hour_average",
            "best_iteration": final_model.booster.best_iteration
            or final_model.booster.current_iteration(),
        }
    ]
)
overall_results.to_csv(OUTPUT_DIR / "rmse_overall.csv", index=False)

# Per-station RMSE on the last predicted week, using weekly mean actual vs weekly mean forecast.
station_results_df = compute_last_week_station_rmse_results(test_output)
station_results_df.to_csv(OUTPUT_DIR / "rmse_by_station.csv", index=False)

importance = pd.DataFrame(
    {
        "feature": final_model.booster.feature_name(),
        "importance_gain": final_model.booster.feature_importance(importance_type="gain"),
        "importance_split": final_model.booster.feature_importance(importance_type="split"),
    }
).sort_values("importance_gain", ascending=False)
importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

print("Overall test results")
print(overall_results)
print("Station-level test results")
print(station_results_df)


# %% [markdown]
# ## Evaluation and Interpretability Figures

# %%


def walk_forward_validation_subtitle(results: pd.DataFrame) -> str:
    valid_start = pd.to_datetime(results["valid_start"].min())
    valid_end = pd.to_datetime(results["valid_end"].max())
    return (
        f"Expanding-window, {len(results)} weekly folds: train on all prior data, "
        f"score next unseen week (~{int(results['n'].median())} rows).\n"
        f"Validation weeks {valid_start:%Y-%m-%d} to {valid_end:%Y-%m-%d}."
    )


def save_validation_figure(results: pd.DataFrame) -> None:
    train_cutoff = pd.to_datetime(results["end_window"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        train_cutoff,
        results["rmse"],
        marker="o",
        label="LightGBM",
        color=SLIDE_COLORS["accent"],
    )
    ax.plot(
        train_cutoff,
        results["baseline_rmse"],
        marker="o",
        label="Naive baseline",
        color=SLIDE_COLORS["contrast"],
    )
    layout_top = style_figure_title(
        ax, "Walk-forward validation RMSE", walk_forward_validation_subtitle(results)
    )
    ax.set_xlabel("Training data through (local time)")
    ax.set_ylabel("RMSE")
    ax.set_xticks(train_cutoff)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    style_axis(ax)
    style_legend(ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))
    save_figure(fig, FIGURE_DIR / "walk_forward_validation.png")
    save_figure_csv("walk_forward_validation", results.set_index("end_window"))
    plt.close(fig)


def save_station_rmse_figure(
    results: pd.DataFrame,
    train_cutoff_local: pd.Timestamp,
) -> None:
    plot_df = results.sort_values("rmse", ascending=True)
    y = np.arange(len(plot_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(y - width / 2, plot_df["rmse"], width, label="LightGBM", color=SLIDE_COLORS["accent"])
    ax.barh(
        y + width / 2,
        plot_df["baseline_rmse"],
        width,
        label="Baseline",
        color=SLIDE_COLORS["contrast"],
    )
    subtitle = station_rmse_figure_subtitle(
        results["last_predicted_week_start"].iloc[0],
        results["last_predicted_week_end"].iloc[0],
        results["n"],
        train_cutoff_local,
    )
    layout_top = style_figure_title(ax, "Test RMSE by station", subtitle)
    ax.set_xlabel("RMSE")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["station_display_name"])
    style_axis(ax, grid_axis="x")
    style_legend(ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))
    save_figure(fig, FIGURE_DIR / "rmse_by_station.png")
    save_figure_csv("rmse_by_station", plot_df.reset_index(drop=True))
    plt.close(fig)


def save_prediction_figure(output: pd.DataFrame) -> None:
    station_name = (
        output.groupby("station_name", observed=True)["actual_passenger_count"].sum().idxmax()
    )
    plot_df = output[output["station_name"] == station_name].sort_values("predicted_for_local")
    station_display_name = plot_df["station_display_name"].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        plot_df["predicted_for_local"],
        plot_df["actual_passenger_count"],
        marker="o",
        label="Actual",
        color=SLIDE_COLORS["text"],
    )
    ax.plot(
        plot_df["predicted_for_local"],
        plot_df["predicted_passenger_count"],
        marker="o",
        label="LightGBM",
        color=SLIDE_COLORS["accent"],
    )
    ax.plot(
        plot_df["predicted_for_local"],
        plot_df["baseline_predicted_passenger_count"],
        marker="o",
        label="Baseline",
        color=SLIDE_COLORS["contrast"],
    )
    ax.set_title(f"Test-week forecast profile: {station_display_name}")
    ax.set_xlabel("Target timestamp")
    ax.set_ylabel("Passengers")
    style_axis(ax)
    style_legend(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "test_prediction_vs_actual.png")
    save_figure_csv("test_prediction_vs_actual", plot_df.reset_index(drop=True))
    plt.close(fig)


def save_hourly_scatter_figure(output: pd.DataFrame) -> None:
    plot_df = output[output["actual_passenger_count"].notna()].copy()
    plot_df["hour"] = pd.to_datetime(plot_df["predicted_for_local"]).dt.hour

    series = [
        ("Actual", "actual_passenger_count", -0.16, SLIDE_COLORS["text"]),
        ("LightGBM", "predicted_passenger_count", 0.0, SLIDE_COLORS["accent"]),
        ("Naive baseline", "baseline_predicted_passenger_count", 0.16, SLIDE_COLORS["contrast"]),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, column, offset, color in series:
        ax.scatter(
            plot_df["hour"] + offset,
            plot_df[column],
            s=28,
            alpha=0.55,
            label=label,
            color=color,
            edgecolors="none",
        )

    ax.set_title("Passenger count by hour: actual vs forecast")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Passengers")
    ax.set_xticks(sorted(plot_df["hour"].unique()))
    style_axis(ax)
    style_legend(ax)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "hourly_scatter_actual_vs_predictions.png")
    save_figure_csv("hourly_scatter_actual_vs_predictions", plot_df.reset_index(drop=True))
    plt.close(fig)


def save_station_scatter_figure(
    output: pd.DataFrame,
    train_cutoff_local: pd.Timestamp,
) -> None:
    plot_df = output[output["actual_passenger_count"].notna()].copy()
    plot_df = plot_df[plot_df["predicted_for_local"] == STATION_SNAPSHOT_PREDICTED_FOR_LOCAL].copy()
    station_order = (
        plot_df.groupby("station_display_name", observed=True)["actual_passenger_count"]
        .median()
        .sort_values()
        .index.tolist()
    )
    station_positions = {station: idx for idx, station in enumerate(station_order)}
    plot_df["station_position"] = plot_df["station_display_name"].map(station_positions)

    series = [
        ("Actual", "actual_passenger_count", -0.22, SLIDE_COLORS["text"]),
        ("LightGBM", "predicted_passenger_count", 0.0, SLIDE_COLORS["accent"]),
        ("Naive baseline", "baseline_predicted_passenger_count", 0.22, SLIDE_COLORS["contrast"]),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, column, offset, color in series:
        ax.scatter(
            plot_df["station_position"] + offset,
            plot_df[column],
            s=28,
            label=label,
            color=color,
            edgecolors="none",
        )

    layout_top = style_figure_title(
        ax, "Passenger count by station", station_snapshot_subtitle(train_cutoff_local)
    )
    ax.set_xlabel("Station")
    ax.set_ylabel("Passengers")
    ax.set_xticks(range(len(station_order)))
    ax.set_xticklabels(station_order, rotation=35, ha="right")
    style_axis(ax)
    style_legend(ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))
    save_figure(fig, FIGURE_DIR / "station_scatter_actual_vs_predictions.png")
    save_figure_csv("station_scatter_actual_vs_predictions", plot_df.reset_index(drop=True))
    plt.close(fig)


def save_feature_importance_figure(importance_df: pd.DataFrame) -> None:
    plot_df = importance_df.head(12).sort_values("importance_gain")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature"], plot_df["importance_gain"], color=SLIDE_COLORS["accent"])
    ax.set_title("LightGBM feature importance")
    ax.set_xlabel("Gain")
    style_axis(ax, grid_axis="x")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "feature_importance.png")
    save_figure_csv("feature_importance", importance_df.reset_index(drop=True))
    plt.close(fig)


def save_shap_summary_figure(model: TrainedModel, reference: pd.DataFrame) -> None:
    sampled_reference = reference.dropna(subset=[TARGET]).sample(
        min(2000, reference[TARGET].notna().sum()),
        random_state=RANDOM_SEED,
    )
    filled_reference = sampled_reference.copy()
    for column, value in model.fill_values.items():
        filled_reference[column] = filled_reference[column].fillna(value)

    shap_values = model.booster.predict(
        as_model_matrix(filled_reference),
        pred_contrib=True,
        num_iteration=model.booster.best_iteration or model.booster.current_iteration(),
    )
    shap_df = pd.DataFrame(shap_values, columns=[*FEATURE_COLUMNS, "base_value"])
    mean_abs_shap = (
        shap_df[FEATURE_COLUMNS]
        .abs()
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(mean_abs_shap.index, mean_abs_shap.values, color=SLIDE_COLORS["contrast"])
    ax.set_title("Mean absolute SHAP contribution")
    ax.set_xlabel("Average absolute contribution to prediction")
    style_axis(ax, grid_axis="x")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "shap_summary.png")
    save_figure_csv(
        "shap_summary",
        mean_abs_shap.rename("mean_absolute_shap_contribution").rename_axis("feature"),
    )
    plt.close(fig)


def partial_dependence(
    model: TrainedModel,
    reference: pd.DataFrame,
    feature: str,
    grid: np.ndarray,
) -> pd.DataFrame:
    rows = []
    sample = reference.sample(min(2000, len(reference)), random_state=RANDOM_SEED).copy()
    for value in grid:
        scenario = sample.copy()
        scenario[feature] = value
        rows.append(
            {
                "feature": feature,
                "value": value,
                "prediction": predict_lightgbm(model, scenario).mean(),
            }
        )
    return pd.DataFrame(rows)


def save_partial_dependence_figures(
    model: TrainedModel,
    reference: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> None:
    candidate_features = [
        feature
        for feature in importance_df["feature"]
        if feature in [*NUMERIC_FEATURES, "hour", "month"]
    ][:4]
    if not candidate_features:
        return

    fig, axes = plt.subplots(1, len(candidate_features), figsize=(5 * len(candidate_features), 4))
    if len(candidate_features) == 1:
        axes = [axes]

    partial_dependence_frames = []

    for axis, feature in zip(axes, candidate_features, strict=True):
        if feature in CATEGORICAL_FEATURES:
            grid = np.asarray(sorted(reference[feature].dropna().unique()))
        else:
            low = reference[feature].quantile(0.05)
            high = reference[feature].quantile(0.95)
            grid = np.linspace(low, high, 12)
        pdp = partial_dependence(model, reference, feature, grid)
        partial_dependence_frames.append(pdp)
        axis.plot(pdp["value"], pdp["prediction"], marker="o", color=SLIDE_COLORS["accent"])
        axis.set_title(feature)
        axis.set_xlabel(feature)
        axis.set_ylabel("Average prediction")
        style_axis(axis)

    fig.suptitle("Manual partial dependence for top features", y=1.03)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "partial_dependence_top_features.png", bbox_inches="tight")
    save_figure_csv(
        "partial_dependence_top_features",
        pd.concat(partial_dependence_frames, ignore_index=True),
    )
    plt.close(fig)


save_validation_figure(walk_forward_results)
save_station_rmse_figure(station_results_df, train_frame["local_timestamp"].max())
save_prediction_figure(test_output)
save_hourly_scatter_figure(test_output)
save_station_scatter_figure(test_output, train_frame["local_timestamp"].max())
save_feature_importance_figure(importance)
save_shap_summary_figure(final_model, train_frame)
save_partial_dependence_figures(final_model, train_frame, importance)


# %% [markdown]
# ## Executive Readout
#
# LightGBM is compared against a strong seasonal naive baseline. The baseline answers, "what was the
# average passenger count at this station and hour across the previous four weeks?" The model should
# beat that by exploiting station
# metadata, traffic, weather, holiday/event flags, and recent demand lags. If it does not beat the
# baseline, the production recommendation would be to ship the baseline first and keep the model in
# shadow mode.

# %%

print("Artifacts written:")
for path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"- {path.relative_to(BASE_DIR)}")
for path in sorted(FIGURE_DIR.glob("*.png")):
    print(f"- {path.relative_to(BASE_DIR)}")
