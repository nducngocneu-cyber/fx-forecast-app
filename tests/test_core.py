"""Unit tests. Run from the repo root:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import dm_test, rolling_backtest, summarize
from src.data_loader import clean_spikes, load_fx_data, simulate_usdvnd, to_business_days
from src.models import (MODEL_REGISTRY, ARIMAForecaster, GBMForecaster, RandomWalkForecaster,
                        make_features)


@pytest.fixture(scope="module")
def series():
    return simulate_usdvnd(start="2021-01-01", end="2024-12-31", seed=7)


def test_simulation_is_reproducible_and_positive(series):
    again = simulate_usdvnd(start="2021-01-01", end="2024-12-31", seed=7)
    pd.testing.assert_series_equal(series, again)
    assert (series > 0).all()
    assert series.index.freqstr == "B"


def test_clean_spikes_removes_reverting_tick_but_keeps_level_shift():
    idx = pd.bdate_range("2024-01-01", periods=60)
    s = pd.Series(25_000.0, index=idx)
    s.iloc[20] = 25_400.0        # one-day spike (+1.6%) that reverts
    s.iloc[40:] = 25_300.0       # persistent shift (+1.2%)
    cleaned, flagged = clean_spikes(s)
    assert flagged.iloc[20] and cleaned.iloc[20] == 25_000.0
    assert cleaned.iloc[45:].eq(25_300.0).all()


def test_business_day_reindex_fills_gaps():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-05"]))
    out = to_business_days(s)
    assert len(out) == 5 and out.iloc[3] == 1.0


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_models_return_ordered_intervals(series, name):
    fc = MODEL_REGISTRY[name]().fit(series.iloc[-500:]).predict(10, level=0.9)
    assert list(fc.columns) == ["point", "lower", "upper"]
    assert len(fc) == 10 and fc.index[0] > series.index[-500:][-1]
    assert (fc["lower"] <= fc["point"]).all() and (fc["point"] <= fc["upper"]).all()
    width = fc["upper"] - fc["lower"]
    assert width.iloc[-1] > width.iloc[0]          # uncertainty grows with horizon


def test_wider_level_gives_wider_band(series):
    m = RandomWalkForecaster().fit(series)
    w80 = m.predict(5, 0.8).eval("upper - lower")
    w95 = m.predict(5, 0.95).eval("upper - lower")
    assert (w95 > w80).all()


def test_arima_fixed_order_is_respected(series):
    m = ARIMAForecaster(order=(1, 1, 0), trend="t").fit(series.iloc[-400:])
    assert m.order_ == (1, 1, 0) and m.trend_ == "t"


def test_gbm_features_match_between_training_and_forecast(series):
    lp = np.log(series.to_numpy()[-100:])
    win = np.lib.stride_tricks.sliding_window_view(lp, 21)
    assert np.allclose(make_features(win)[-1], make_features(lp[-21:][None, :])[0])


def test_gbm_is_deterministic(series):
    a = GBMForecaster(seed=1).fit(series.iloc[-400:]).predict(5)
    b = GBMForecaster(seed=1).fit(series.iloc[-400:]).predict(5)
    pd.testing.assert_frame_equal(a, b)


def test_dm_test_detects_clear_difference():
    rng = np.random.default_rng(0)
    stat, p = dm_test(-1 - rng.random(60), h=1, step=1)   # model always better
    assert stat < 0 and p < 0.01
    stat, p = dm_test(rng.standard_normal(60), h=1, step=1)
    assert p > 0.01


def test_backtest_has_no_lookahead_and_summarises(series):
    bt = rolling_backtest(series, train_size=300, horizon=5, n_origins=4, step=5)
    # actual at h equals the observed price h business days after the origin
    row = bt.iloc[0]
    pos = series.index.get_loc(row["origin"])
    assert row["actual"] == series.iloc[pos + int(row["h"])]
    summ = summarize(bt, step=5)
    rw = summ[summ["model"] == "Random walk"]
    assert np.allclose(rw["rel_RMSE"], 1.0) and rw["hit_rate"].isna().all()


def test_loader_falls_back_to_simulated_label():
    d = load_fx_data("simulated")
    assert d.is_simulated and "Simulated" in d.label
