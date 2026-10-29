"""VND government bond portfolio: carry and roll-down versus curve risk.

Universe: one par bond per published tenor, priced off the calibrated VND zero curve.

* **Expected return** over the horizon = carry + roll-down *if the curve does not move*:
  coupon income plus the price gain from the bond "rolling down" to a shorter maturity
  on today's curve. This is the desk's standard neutral-scenario return, not a forecast.
* **Risk** = P&L volatility from curve moves, using key-rate DV01s and the historical
  covariance of monthly yield changes (VBMA data), scaled to the horizon.
* **Optimisation** (SciPy SLSQP): maximise expected return minus a risk penalty, or trace
  the efficient frontier, under long-only, concentration and duration limits.
* **Stress tests**: full revaluation of the portfolio under curve scenarios, including the
  worst 12-month move in the sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.curves import ZeroCurve, scenario_shift
from src.pricing import Bond, par_bond


@dataclass(frozen=True)
class Universe:
    settle: pd.Timestamp
    horizon_y: float
    table: pd.DataFrame          # one row per bond: tenor, coupon, price, carry/roll, duration …
    krd: pd.DataFrame            # key-rate DV01 per 100 market value (rows bonds, cols tenors)
    bonds: dict                  # tenor -> Bond


def horizon_value(bond: Bond, settle: pd.Timestamp, horizon_y: float, curve: ZeroCurve) -> float:
    """Value at the horizon of holding the bond, curve unchanged: remaining bond priced on
    today's curve at the horizon date + coupons received, reinvested to the horizon."""
    h_date = settle + pd.DateOffset(months=int(round(horizon_y * 12)))
    cf = bond.cashflows(settle)
    received = cf[cf["date"] <= h_date]
    reinvest = np.sum(received["cf"].to_numpy()
                      * np.exp(curve.zero(np.maximum(horizon_y - received["t"].to_numpy(), 1e-6))
                               * (horizon_y - received["t"].to_numpy())))
    remaining = bond.dirty_from_curve(h_date, curve) if bond.maturity > h_date else 0.0
    return float(remaining + reinvest)


def build_universe(curve: ZeroCurve, settle: pd.Timestamp, tenors, horizon_y: float = 1.0) -> Universe:
    rows, krd, bonds = [], {}, {}
    for T in tenors:
        b = par_bond(settle, T, curve, freq=1)
        price = b.dirty_from_curve(settle, curve)
        hv = horizon_value(b, settle, horizon_y, curve)
        risk = b.risk_from_yield(settle, b.ytm(settle, price - b.accrued(settle)))
        kr = b.key_rate_dv01(settle, curve, tenors) / price * 100          # per 100 market value
        carry = b.coupon_pct * horizon_y
        total = (hv / price - 1) * 100
        rows.append({"tenor": T, "coupon_pct": b.coupon_pct, "price": price,
                     "carry_pct": carry, "rolldown_pct": total - carry, "exp_return_pct": total,
                     "mod_duration": risk["modified"], "dv01_per_100": kr.sum()})
        krd[T] = kr
        bonds[T] = b
    return Universe(settle, horizon_y, pd.DataFrame(rows).set_index("tenor"),
                    pd.DataFrame(krd).T, bonds)


def yield_change_cov(yields: pd.DataFrame, horizon_y: float) -> pd.DataFrame:
    """Covariance (bp²) of monthly yield changes, scaled to the horizon assuming i.i.d. months."""
    d = yields.diff().dropna() * 100
    return d.cov() * (12 * horizon_y)


def portfolio_stats(w: np.ndarray, u: Universe, cov_bp: pd.DataFrame) -> dict:
    w = np.asarray(w, float)
    exp_ret = float(w @ u.table["exp_return_pct"].to_numpy())
    K = u.krd[cov_bp.columns].to_numpy()                # % return per 1 bp at each key tenor (DV01 per 100 MV)
    exposure = w @ K                                    # % return per bp, by key tenor
    var = float(exposure @ cov_bp.to_numpy() @ exposure)
    dur = float(w @ u.table["mod_duration"].to_numpy())
    return {"exp_return_pct": exp_ret, "vol_pct": np.sqrt(var), "duration": dur,
            "dv01_per_100": float(w @ u.table["dv01_per_100"].to_numpy())}


def optimise(u: Universe, cov_bp: pd.DataFrame, risk_aversion: float = 2.0,
             max_weight: float = 0.4, max_duration: float | None = None,
             target_vol: float | None = None, mode: str = "utility",
             x0: np.ndarray | None = None) -> np.ndarray:
    """Long-only weights summing to 1, each at most ``max_weight``.

    mode="utility":    maximise E[r] − (λ/2)·σ²          (λ = ``risk_aversion``)
    mode="min_risk":   minimise σ²
    mode="max_return": maximise E[r]
    ``target_vol`` adds the constraint σ ≤ target (used to trace the frontier).
    """
    n = len(u.table)
    mu = u.table["exp_return_pct"].to_numpy()
    dur = u.table["mod_duration"].to_numpy()
    K = u.krd[cov_bp.columns].to_numpy()
    S = K @ cov_bp.to_numpy() @ K.T
    if max_weight * n < 1 - 1e-9:
        raise ValueError(f"Weight cap {max_weight:.0%} × {n} bonds cannot sum to 100%.")

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]
    if max_duration is not None:
        cons.append({"type": "ineq", "fun": lambda w: max_duration - w @ dur, "jac": lambda w: -dur})
    if target_vol is not None:
        cons.append({"type": "ineq", "fun": lambda w: target_vol**2 - w @ S @ w,
                     "jac": lambda w: -2 * S @ w})
    if mode == "min_risk":
        obj, jac = (lambda w: w @ S @ w), (lambda w: 2 * S @ w)
    elif mode == "max_return" or target_vol is not None:
        obj, jac = (lambda w: -(w @ mu)), (lambda w: -mu)
    else:
        obj = lambda w: -(w @ mu) + 0.5 * risk_aversion * (w @ S @ w)
        jac = lambda w: -mu + risk_aversion * S @ w

    bounds = [(0.0, max_weight)] * n
    starts = [x0] if x0 is not None else []
    tilt = np.clip(mu - mu.min() + 1e-3, 0, None)
    starts += [np.full(n, 1 / n), tilt / tilt.sum(), np.full(n, 1 / n)[::-1]]
    best = None
    for start in starts:
        res = minimize(obj, np.clip(start, 0, max_weight), jac=jac, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
        feasible = all((c["fun"](res.x) >= -1e-7) if c["type"] == "ineq" else abs(c["fun"](res.x)) < 1e-6
                       for c in cons)
        if feasible and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise ValueError("No feasible portfolio: loosen the duration or weight limits.")
    w = np.clip(best.x, 0, None)
    return w / w.sum()


def efficient_frontier(u: Universe, cov_bp: pd.DataFrame, max_weight: float = 0.4,
                       max_duration: float | None = None, n_points: int = 25) -> pd.DataFrame:
    """Highest carry+roll for each risk level, between the min-risk and max-return portfolios."""
    kw = dict(max_weight=max_weight, max_duration=max_duration)
    w_min = optimise(u, cov_bp, mode="min_risk", **kw)
    w_max = optimise(u, cov_bp, mode="max_return", **kw)
    v_lo = portfolio_stats(w_min, u, cov_bp)["vol_pct"]
    v_hi = portfolio_stats(w_max, u, cov_bp)["vol_pct"]
    rows = []
    for v in np.linspace(v_lo, v_hi, n_points):
        try:
            w = optimise(u, cov_bp, target_vol=v * (1 + 1e-6), x0=w_min, **kw)
        except ValueError:
            continue
        rows.append({**portfolio_stats(w, u, cov_bp),
                     **{f"w_{int(t) if float(t).is_integer() else t}Y": x for t, x in zip(u.table.index, w)}})
    return pd.DataFrame(rows).drop_duplicates(subset=["vol_pct"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stress tests
# --------------------------------------------------------------------------- #
def standard_scenarios(yields: pd.DataFrame) -> dict[str, dict[float, float]]:
    """Curve shocks in bp by tenor. The historical one is the largest 12-month rise in the
    10Y yield in the sample, applied tenor by tenor."""
    tenors = list(yields.columns)
    scen = {
        "Parallel +100 bp": {t: 100 for t in tenors},
        "Parallel −100 bp": {t: -100 for t in tenors},
        "Bear steepener (+25 short / +100 long)": dict(zip(tenors, np.linspace(25, 100, len(tenors)))),
        "Bear flattener (+100 short / +25 long)": dict(zip(tenors, np.linspace(100, 25, len(tenors)))),
        "Bull flattener (−25 short / −100 long)": dict(zip(tenors, np.linspace(-25, -100, len(tenors)))),
    }
    chg12 = (yields - yields.shift(12)).dropna() * 100
    if len(chg12):
        end = chg12[10.0 if 10.0 in chg12.columns else chg12.columns[-1]].idxmax()
        start = end - pd.DateOffset(months=12)
        scen[f"Historical: {start:%b %Y} → {end:%b %Y}"] = chg12.loc[end].to_dict()
    return scen


def stress_test(w, u: Universe, curve: ZeroCurve, scenarios: dict) -> pd.DataFrame:
    """Instantaneous full revaluation (% of portfolio value) under each scenario."""
    w = np.asarray(w, float)
    base = np.array([u.bonds[t].dirty_from_curve(u.settle, curve) for t in u.table.index])
    rows = []
    for name, pts in scenarios.items():
        shocked = curve.shifted(scenario_shift(pts))
        new = np.array([u.bonds[t].dirty_from_curve(u.settle, shocked) for t in u.table.index])
        pnl = (new / base - 1) * 100
        rows.append({"scenario": name, "portfolio_pnl_pct": float(w @ pnl),
                     **{f"{int(t) if float(t).is_integer() else t}Y": p for t, p in zip(u.table.index, pnl)}})
    return pd.DataFrame(rows)
