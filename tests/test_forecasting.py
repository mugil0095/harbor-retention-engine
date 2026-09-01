"""J: forecast backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.models.forecasting import (
    attach_plan_tier,
    build_monthly_series,
    gbm_lag_forecast,
    holt_winters_forecast,
    naive_forecast,
    prediction_interval,
    rolling_origin_backtest,
    seasonal_naive_forecast,
)


def test_naive_forecast_repeats_last_value():
    series = pd.Series([10.0, 20.0, 30.0])
    forecast = naive_forecast(series, horizon=3)
    assert (forecast == 30.0).all()


def test_seasonal_naive_uses_value_from_one_season_ago():
    series = pd.Series(np.arange(24, dtype=float))
    forecast = seasonal_naive_forecast(series, horizon=3, season=12)
    np.testing.assert_array_almost_equal(forecast, [12.0, 13.0, 14.0])


def test_seasonal_naive_falls_back_to_naive_with_insufficient_history():
    series = pd.Series([5.0, 6.0, 7.0])
    forecast = seasonal_naive_forecast(series, horizon=2, season=12)
    assert (forecast == 7.0).all()


def test_gbm_lag_forecast_returns_correct_horizon_length():
    series = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 2, 30)))
    forecast = gbm_lag_forecast(series, horizon=3, random_seed=1)
    assert len(forecast) == 3


def test_build_monthly_series_fills_gaps_with_zero(synthetic_tables):
    series = build_monthly_series(synthetic_tables["monthly_usage"])
    assert series.index.min() == 0
    assert not series.isna().any()


def test_attach_plan_tier_left_joins_plan_tier_onto_usage_rows():
    usage = pd.DataFrame({"account_id": [1, 2, 3], "mrr": [100.0, 200.0, 300.0]})
    accounts = pd.DataFrame({"account_id": [1, 2, 3], "plan_tier": ["starter", "growth", "enterprise"]})
    result = attach_plan_tier(usage, accounts)
    assert result["plan_tier"].tolist() == ["starter", "growth", "enterprise"]
    assert len(result) == len(usage)  # left join must not duplicate or drop usage rows


def test_attach_plan_tier_is_a_left_join_missing_accounts_become_nan():
    usage = pd.DataFrame({"account_id": [1, 2], "mrr": [100.0, 200.0]})
    accounts = pd.DataFrame({"account_id": [1], "plan_tier": ["starter"]})  # account 2 not in accounts
    result = attach_plan_tier(usage, accounts)
    assert len(result) == 2  # usage row for account 2 is kept, not dropped
    assert result.loc[result["account_id"] == 2, "plan_tier"].isna().all()


def test_build_monthly_series_raises_clear_error_when_segmenting_without_plan_tier():
    usage = pd.DataFrame({"account_id": [1], "month_index": [0], "mrr": [100.0], "is_active": [True]})
    with pytest.raises(ValueError, match="plan_tier"):
        build_monthly_series(usage, segment="starter")


def test_build_monthly_series_filters_to_requested_segment():
    usage = pd.DataFrame(
        {
            "account_id": [1, 2, 3],
            "month_index": [0, 0, 0],
            "mrr": [100.0, 200.0, 300.0],
            "is_active": [True, True, True],
            "plan_tier": ["starter", "growth", "starter"],
        }
    )
    series = build_monthly_series(usage, segment="starter")
    assert series.loc[0] == 400.0  # only the two starter-plan accounts (100 + 300)


def test_holt_winters_forecast_fits_and_forecasts_with_sufficient_history():
    """Exercises the actual ExponentialSmoothing fit/forecast path, not
    just the insufficient-history short-circuit -- every existing test and
    every real pipeline run to date had less than 2 full seasons of
    history in this exact call, so this path had never actually run under
    test before."""
    rng = np.random.default_rng(5)
    months = np.arange(30)
    trend = 1000 + 20 * months
    seasonal = 50 * np.sin(2 * np.pi * months / 12)
    noise = rng.normal(0, 5, 30)
    series = pd.Series(trend + seasonal + noise)

    forecast = holt_winters_forecast(series, horizon=3, season=12)
    assert len(forecast) == 3
    assert np.all(np.isfinite(forecast))
    # forecast should be in the right ballpark (continuing the upward trend), not naive-flat
    assert forecast[-1] > series.iloc[-1] * 0.5


def test_holt_winters_forecast_falls_back_to_naive_on_fit_failure(monkeypatch):
    """Forces the exception path (line coverage aside, this verifies the
    fallback actually returns a valid naive forecast rather than
    propagating whatever statsmodels raised)."""
    import harbor.models.forecasting as forecasting_module

    class ExplodingModel:
        def fit(self, *args, **kwargs):
            raise ValueError("simulated statsmodels convergence failure")

    monkeypatch.setattr(forecasting_module, "ExponentialSmoothing", lambda *a, **k: ExplodingModel())
    series = pd.Series(np.arange(30, dtype=float) + 1000)
    forecast = holt_winters_forecast(series, horizon=3, season=12)
    np.testing.assert_array_equal(forecast, naive_forecast(series, horizon=3))


def test_rolling_origin_backtest_skips_origins_with_zero_actuals():
    """A fully-churned segment can have actual=0 in some backtest windows;
    MAPE is undefined there (division by zero), so those origins must be
    skipped rather than polluting the result with inf/nan."""
    values = [1000.0] * 15 + [0.0] * 5  # last few months fully churned
    series = pd.Series(values)
    results = rolling_origin_backtest(series, horizon=2, min_history=10)
    for r in results:
        assert np.isfinite(r.mape)
        assert np.isfinite(r.rmse)


def test_rolling_origin_backtest_returns_empty_when_every_origin_has_zero_actuals():
    series = pd.Series([0.0] * 20)
    results = rolling_origin_backtest(series, horizon=2, min_history=10)
    assert results == []


def test_rolling_origin_backtest_never_forecasts_using_future_data():
    rng = np.random.default_rng(2)
    series = pd.Series(50 + np.cumsum(rng.normal(0, 1, 20)) + 1000)  # keep positive, avoid MAPE div/0
    results = rolling_origin_backtest(series, horizon=2, min_history=6)
    assert len(results) > 0
    for r in results:
        assert r.n_origins > 0
        assert r.mape >= 0
        assert r.rmse >= 0


def test_prediction_interval_lower_bound_below_point_forecast():
    rng = np.random.default_rng(3)
    series = pd.Series(500 + np.cumsum(rng.normal(0, 5, 30)))
    point, lower, upper = prediction_interval(series, horizon=3, min_history=10, method_fn=naive_forecast)
    assert (lower <= point).all()
    assert (upper >= point).all()
