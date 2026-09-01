"""MRR forecasting: naive baselines -> classical statistical model ->
gradient boosting with lag features, evaluated with rolling-origin
backtesting (never a single train/test split on a time series).

Forecasts are produced at the whole-portfolio level and per plan-tier
segment, since a retention team cares about revenue-at-risk by segment, not
just an aggregate number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

LAGS = (1, 2, 3, 6, 12)


def attach_plan_tier(monthly_usage: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    return monthly_usage.merge(accounts[["account_id", "plan_tier"]], on="account_id", how="left")


def build_monthly_series(monthly_usage: pd.DataFrame, segment: str | None = None) -> pd.Series:
    df = monthly_usage
    if segment is not None:
        if "plan_tier" not in df.columns:
            raise ValueError("monthly_usage must have plan_tier attached (see attach_plan_tier) to segment.")
        df = df[df["plan_tier"] == segment]
    series = df[df["is_active"]].groupby("month_index")["mrr"].sum()
    full_index = pd.RangeIndex(series.index.min(), series.index.max() + 1)
    return series.reindex(full_index).fillna(0.0)


def naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(history.iloc[-1], horizon)


def seasonal_naive_forecast(history: pd.Series, horizon: int, season: int = 12) -> np.ndarray:
    if len(history) < season:
        return naive_forecast(history, horizon)
    return np.array([history.iloc[-season + (h % season)] for h in range(horizon)])


def holt_winters_forecast(history: pd.Series, horizon: int, season: int = 12) -> np.ndarray:
    if len(history) < 2 * season:
        return naive_forecast(history, horizon)
    try:
        model = ExponentialSmoothing(
            history.to_numpy(), trend="add", seasonal="add", seasonal_periods=season
        ).fit(optimized=True)
        return model.forecast(horizon)
    except Exception:  # noqa: BLE001 -- statsmodels can raise several unrelated error types
        # (LinAlgError, ValueError from a degenerate series, etc.) on real
        # data; any of them should fall back to naive, not crash the backtest.
        return naive_forecast(history, horizon)


def _make_lag_frame(history: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": history.to_numpy()})
    for lag in LAGS:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["month_of_year"] = np.arange(len(df)) % 12
    return df


def gbm_lag_forecast(history: pd.Series, horizon: int, random_seed: int = 42) -> np.ndarray:
    df = _make_lag_frame(history).dropna()
    feature_cols = [f"lag_{lag}" for lag in LAGS] + ["rolling_mean_3", "month_of_year"]
    if len(df) < 10:
        return naive_forecast(history, horizon)

    model = HistGradientBoostingRegressor(random_state=random_seed, max_depth=3, max_iter=150)
    model.fit(df[feature_cols], df["y"])

    extended = list(history.to_numpy())
    preds = []
    for h in range(horizon):
        row = {f"lag_{lag}": extended[-lag] for lag in LAGS if lag <= len(extended)}
        for lag in LAGS:
            row.setdefault(f"lag_{lag}", extended[0])
        row["rolling_mean_3"] = float(np.mean(extended[-3:]))
        row["month_of_year"] = (len(extended)) % 12
        x = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(x)[0])
        preds.append(pred)
        extended.append(pred)
    return np.array(preds)


FORECAST_METHODS = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "holt_winters": holt_winters_forecast,
    "gbm_lags": gbm_lag_forecast,
}


@dataclass
class BacktestResult:
    method: str
    mape: float
    rmse: float
    n_origins: int


def rolling_origin_backtest(
    series: pd.Series, horizon: int, min_history: int, methods: dict | None = None
) -> list[BacktestResult]:
    methods = methods or FORECAST_METHODS
    errors: dict[str, list[float]] = {name: [] for name in methods}
    sq_errors: dict[str, list[float]] = {name: [] for name in methods}
    n_origins = 0

    for origin in range(min_history, len(series) - horizon + 1):
        history = series.iloc[:origin]
        actual = series.iloc[origin : origin + horizon].to_numpy()
        if np.any(actual == 0):
            continue  # avoid divide-by-zero in MAPE for fully-churned segments
        n_origins += 1
        for name, fn in methods.items():
            pred = fn(history, horizon)
            ape = np.abs((actual - pred) / actual)
            errors[name].extend(ape.tolist())
            sq_errors[name].extend(((actual - pred) ** 2).tolist())

    results = []
    for name in methods:
        if not errors[name]:
            continue
        results.append(
            BacktestResult(
                method=name,
                mape=float(np.mean(errors[name])),
                rmse=float(np.sqrt(np.mean(sq_errors[name]))),
                n_origins=n_origins,
            )
        )
    return results


def prediction_interval(
    series: pd.Series, horizon: int, min_history: int, method_fn, z: float = 1.645
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual-based approximate prediction interval from backtest errors.
    Returns (point_forecast, lower, upper) for a forecast made from the full series."""
    residuals_by_step: list[list[float]] = [[] for _ in range(horizon)]
    for origin in range(min_history, len(series) - horizon + 1):
        history = series.iloc[:origin]
        actual = series.iloc[origin : origin + horizon].to_numpy()
        pred = method_fn(history, horizon)
        for h in range(horizon):
            residuals_by_step[h].append(actual[h] - pred[h])

    point = method_fn(series, horizon)
    stds = np.array(
        [np.std(r) if len(r) > 1 else 0.0 for r in residuals_by_step]
    )
    return point, point - z * stds, point + z * stds
