"""Forecasting the VND government yield curve: dynamic Nelson–Siegel (Diebold and Li, 2006).

Each month the curve is summarised by three Nelson–Siegel factors – level, slope and
curvature – estimated by OLS on the published yields with the decay parameter λ fixed.
Forecasts of the factors map back to forecasts of every tenor.

Models compared out of sample (expanding window, re-estimated at every origin):
* **Random walk** – no change in any yield. The benchmark to beat.
* **DNS-AR(1)** – each factor follows its own AR(1); direct h-step regression
  β_{t+h} = c + φ·β_t, as in Diebold and Li (2006).
* **AR(1) on yields** – the same direct regression run tenor by tenor, no curve structure.

The sample is short (monthly data from January 2021), so the tests have little power.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import dm_test
from src.curves import ns_loadings

LAMBDA_GRID = np.round(np.arange(0.10, 2.01, 0.05), 2)
MODELS = ["Random walk", "DNS-AR(1)", "AR(1) on yields"]


def dns_factors(yields: pd.DataFrame, lam: float) -> pd.DataFrame:
    """OLS level/slope/curvature for every month (yields in %, maturities in years)."""
    X = ns_loadings(yields.columns.to_numpy(float), lam)
    beta, *_ = np.linalg.lstsq(X, yields.to_numpy().T, rcond=None)
    return pd.DataFrame(beta.T, index=yields.index, columns=["level", "slope", "curvature"])


def choose_lambda(yields: pd.DataFrame) -> float:
    """λ minimising the in-sample fitting RMSE over the given window only."""
    mats = yields.columns.to_numpy(float)
    best, best_rmse = None, np.inf
    for lam in LAMBDA_GRID:
        X = ns_loadings(mats, lam)
        beta, *_ = np.linalg.lstsq(X, yields.to_numpy().T, rcond=None)
        rmse = np.sqrt(np.mean((X @ beta - yields.to_numpy().T) ** 2))
        if rmse < best_rmse:
            best, best_rmse = lam, rmse
    return float(best)


def _direct_ar1(series: np.ndarray, h: int) -> tuple[float, float, np.ndarray]:
    """Fit x_{t+h} = c + φ x_t by OLS; return (c, φ, residuals)."""
    x, y = series[:-h], series[h:]
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0], coef[1], y - A @ coef


def forecast_curve(yields: pd.DataFrame, h: int, lam: float | None = None, level: float = 0.9) -> pd.DataFrame:
    """h-month-ahead forecast of every tenor from each model, with prediction intervals.

    Intervals use the empirical quantiles of the in-sample h-step errors (residuals for the
    AR models, h-month changes for the random walk), so they reflect the fat tails in the data.
    """
    lam = lam or choose_lambda(yields)
    mats = yields.columns.to_numpy(float)
    last = yields.iloc[-1].to_numpy()
    q = [(1 - level) / 2, (1 + level) / 2]
    out = []

    rw_err = (yields.shift(-h) - yields).dropna().to_numpy()
    lo, hi = np.quantile(rw_err, q, axis=0)
    out.append(pd.DataFrame({"model": "Random walk", "tenor": mats, "point": last,
                             "lower": last + lo, "upper": last + hi}))

    F = dns_factors(yields, lam)
    L = ns_loadings(mats, lam)
    f_next, resid = [], []
    for col in F:
        c, phi, e = _direct_ar1(F[col].to_numpy(), h)
        f_next.append(c + phi * F[col].iloc[-1])
        resid.append(e)
    point = L @ np.array(f_next)
    err = np.column_stack(resid) @ L.T
    lo, hi = np.quantile(err, q, axis=0)
    out.append(pd.DataFrame({"model": "DNS-AR(1)", "tenor": mats, "point": point,
                             "lower": point + lo, "upper": point + hi}))

    pts, errs = [], []
    for col in yields:
        c, phi, e = _direct_ar1(yields[col].to_numpy(), h)
        pts.append(c + phi * yields[col].iloc[-1])
        errs.append(e)
    point = np.array(pts)
    lo, hi = np.quantile(np.column_stack(errs), q, axis=0)
    out.append(pd.DataFrame({"model": "AR(1) on yields", "tenor": mats, "point": point,
                             "lower": point + lo, "upper": point + hi}))
    res = pd.concat(out, ignore_index=True)
    res.attrs["lambda"] = lam
    return res


def backtest_curve(yields: pd.DataFrame, horizons=(1, 3, 6), min_train: int = 24,
                   level: float = 0.9) -> pd.DataFrame:
    """Expanding-window backtest. At each origin λ is re-chosen on that window only."""
    rows = []
    n = len(yields)
    for h in horizons:
        for o in range(min_train - 1, n - h):
            train = yields.iloc[: o + 1]
            actual = yields.iloc[o + h].to_numpy()
            fc = forecast_curve(train, h, level=level)
            for model, g in fc.groupby("model", sort=False):
                rows.append(pd.DataFrame({
                    "model": model, "h": h, "origin": train.index[-1], "tenor": g["tenor"].to_numpy(),
                    "point": g["point"].to_numpy(), "lower": g["lower"].to_numpy(),
                    "upper": g["upper"].to_numpy(), "actual": actual,
                    "lambda": fc.attrs["lambda"]}))
    bt = pd.concat(rows, ignore_index=True)
    bt["error_bp"] = (bt["actual"] - bt["point"]) * 100
    return bt


def summarize_curve_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """RMSE (bp) by model, horizon and tenor; relative RMSE and DM p-value vs random walk."""
    g = bt.groupby(["model", "h", "tenor"])
    summ = g.apply(lambda d: pd.Series({
        "RMSE_bp": np.sqrt(np.mean(d["error_bp"] ** 2)),
        "MAE_bp": np.mean(np.abs(d["error_bp"])),
        "coverage": np.mean((d["actual"] >= d["lower"]) & (d["actual"] <= d["upper"])),
        "n": len(d)}), include_groups=False).reset_index()
    bench = summ[summ["model"] == "Random walk"].set_index(["h", "tenor"])["RMSE_bp"]
    summ["rel_RMSE"] = [r / bench.loc[(h, t)] for r, h, t in zip(summ["RMSE_bp"], summ["h"], summ["tenor"])]
    sq = bt.assign(se=bt["error_bp"] ** 2).pivot_table(index=["h", "tenor", "origin"], columns="model", values="se")
    pv = []
    for model, h, t in zip(summ["model"], summ["h"], summ["tenor"]):
        if model == "Random walk":
            pv.append(np.nan)
            continue
        d = sq.loc[(h, t)]
        pv.append(dm_test((d[model] - d["Random walk"]).to_numpy(), h=h, step=1)[1])
    summ["DM_pvalue"] = pv
    return summ
