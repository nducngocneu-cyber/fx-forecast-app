"""Rolling-origin (walk-forward) evaluation.

At each origin t the models are re-estimated on the ``train_size`` business
days ending at t and forecast t+1 … t+H. Nothing after t is used, including
for ARIMA order selection (done once on the first window, then fixed).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.models import ARIMAForecaster, BaseForecaster, GBMForecaster, RandomWalkForecaster

BENCHMARK = "Random walk"


def default_factories(first_window: pd.Series) -> dict[str, Callable[[], BaseForecaster]]:
    """Model constructors for the backtest. ARIMA order is fixed from the first window."""
    pilot = ARIMAForecaster().fit(first_window)
    order, trend = pilot.order_, pilot.trend_
    return {
        "Random walk": lambda: RandomWalkForecaster(),
        "ARIMA": lambda: ARIMAForecaster(order=order, trend=trend),
        "Gradient boosting": lambda: GBMForecaster(n_paths=300),
    }


def rolling_backtest(
    y: pd.Series,
    train_size: int = 756,
    horizon: int = 30,
    n_origins: int = 30,
    step: int = 10,
    level: float = 0.90,
    factories: dict[str, Callable[[], BaseForecaster]] | None = None,
    progress: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """Return one row per (model, origin, h) with forecast, interval and actual."""
    last_origin = len(y) - horizon - 1
    origins = [last_origin - step * k for k in range(n_origins)][::-1]
    origins = [o for o in origins if o >= train_size - 1]
    if not origins:
        raise ValueError("Series too short for the requested backtest.")
    if factories is None:
        factories = default_factories(y.iloc[origins[0] - train_size + 1: origins[0] + 1])

    rows = []
    for i, o in enumerate(origins):
        train = y.iloc[o - train_size + 1: o + 1]
        actual = y.iloc[o + 1: o + 1 + horizon].to_numpy()
        spot = float(train.iloc[-1])
        for name, make in factories.items():
            fc = make().fit(train).predict(horizon, level)
            rows.append(pd.DataFrame({
                "model": name, "origin": train.index[-1], "h": np.arange(1, horizon + 1),
                "spot": spot, "actual": actual,
                "point": fc["point"].to_numpy(), "lower": fc["lower"].to_numpy(),
                "upper": fc["upper"].to_numpy(),
            }))
        if progress:
            progress((i + 1) / len(origins))
    out = pd.concat(rows, ignore_index=True)
    out["error"] = out["actual"] - out["point"]
    return out


def dm_test(loss_diff: np.ndarray, h: int, step: int) -> tuple[float, float]:
    """Diebold–Mariano test with the Harvey–Leybourne–Newbold small-sample correction.

    ``loss_diff`` = squared error of model minus squared error of benchmark, one
    value per origin. Origins are ``step`` days apart, so forecasts overlap when
    h > step; the long-run variance uses ceil(h/step)-1 Newey–West lags.
    Returns (statistic, two-sided p-value); negative statistic = model better.
    """
    d = np.asarray(loss_diff, float)
    n = len(d)
    if n < 5 or np.allclose(d, 0):
        return np.nan, np.nan
    lags = max(int(np.ceil(h / step)) - 1, 0)
    dc = d - d.mean()
    lrv = np.sum(dc**2) / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    if lrv <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(lrv / n)
    hh = lags + 1
    correction = np.sqrt((n + 1 - 2 * hh + hh * (hh - 1) / n) / n)
    stat = dm * correction
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def summarize(bt: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    """Accuracy metrics by model and horizon, relative to the random walk."""
    def _agg(g):
        e = g["error"].to_numpy()
        # Direction of the move vs spot; changes under 0.5 VND count as "no change"
        # (the random walk always predicts no change, so it gets no hit rate).
        def _sign(x):
            return np.where(np.abs(x) < 0.5, 0, np.sign(x))

        move_true = _sign(g["actual"] - g["spot"])
        move_pred = _sign(g["point"] - g["spot"])
        valid = (move_true != 0) & (move_pred != 0)
        return pd.Series({
            "RMSE": np.sqrt(np.mean(e**2)),
            "MAE": np.mean(np.abs(e)),
            "MAPE_%": 100 * np.mean(np.abs(e) / g["actual"]),
            "coverage": np.mean((g["actual"] >= g["lower"]) & (g["actual"] <= g["upper"])),
            "avg_band_width": np.mean(g["upper"] - g["lower"]),
            "hit_rate": np.mean(move_true[valid] == move_pred[valid]) if valid.any() else np.nan,
            "n": len(g),
        })

    summ = bt.groupby(["model", "h"]).apply(_agg, include_groups=False).reset_index()
    bench = summ[summ["model"] == BENCHMARK].set_index("h")["RMSE"]
    summ["rel_RMSE"] = summ["RMSE"] / summ["h"].map(bench)

    # DM test vs random walk at each horizon
    sq = bt.assign(se=bt["error"] ** 2).pivot_table(index=["origin", "h"], columns="model", values="se")
    pvals = []
    for (model, h) in zip(summ["model"], summ["h"]):
        if model == BENCHMARK:
            pvals.append(np.nan)
            continue
        d = sq.xs(h, level="h")
        _, p = dm_test((d[model] - d[BENCHMARK]).to_numpy(), h, step)
        pvals.append(p)
    summ["DM_pvalue"] = pvals
    return summ
