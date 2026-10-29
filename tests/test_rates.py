"""Unit tests for curves, pricing, portfolio, curve forecasting and the SQL warehouse."""
import numpy as np
import pandas as pd
import pytest

from src.curves import ZeroCurve, fit_curve, key_rate_bump, ns_loadings, par_yields_from_curve, scenario_shift
from src.portfolio import (build_universe, efficient_frontier, optimise, portfolio_stats,
                           standard_scenarios, stress_test, yield_change_cov)
from src.pricing import (Bond, cip_forward, cip_forward_money_market, ewma_vol, garman_kohlhagen,
                         implied_vol, par_bond)
from src.rates_data import clean_vn_yields, continuous_tail
from src.rates_forecast import backtest_curve, dns_factors, forecast_curve, summarize_curve_backtest

TENORS = [1, 2, 3, 5, 7, 10, 20, 30]
SETTLE = pd.Timestamp("2026-10-28")


def flat_curve(rate_pct: float) -> ZeroCurve:
    return ZeroCurve(dict(beta0=rate_pct / 100, beta1=0.0, beta2=0.0, beta3=0.0, lam1=1.0, lam2=1.0), "NS")


@pytest.fixture(scope="module")
def synthetic_history():
    """Smooth synthetic curve history (NOT market data) with a level trend and noise."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2021-01-31", periods=48, freq="ME")
    L = ns_loadings(TENORS, 0.5)
    level = 3.5 + np.cumsum(rng.normal(0, 0.1, len(idx)))
    slope = -1.0 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    curv = 0.5 + rng.normal(0, 0.1, len(idx))
    Y = np.column_stack([level, slope, curv]) @ L.T + rng.normal(0, 0.03, (len(idx), len(TENORS)))
    return pd.DataFrame(Y, index=idx, columns=[float(t) for t in TENORS])


# --------------------------------------------------------------------------- curves
def test_flat_curve_par_yield_equals_annual_equivalent_rate():
    c = flat_curve(4.0)
    par = par_yields_from_curve(c.df, [2, 10], freq=1)
    assert np.allclose(par, (np.exp(0.04) - 1) * 100, atol=1e-10)


def test_fit_recovers_known_curve():
    true = ZeroCurve(dict(beta0=0.05, beta1=-0.02, beta2=0.01, beta3=0.0, lam1=0.6, lam2=1.0), "NS")
    obs = true.par_yield(TENORS, 1)
    fit = fit_curve(TENORS, obs, freq=1, model="NS")
    assert fit.rmse_bp < 0.05
    assert np.allclose(fit.zero(np.array(TENORS, float)), true.zero(np.array(TENORS, float)), atol=2e-5)


def test_key_rate_tents_sum_to_parallel_shift():
    t = np.linspace(0.1, 35, 200)
    total = sum(key_rate_bump(k, TENORS)(t) for k in TENORS)
    assert np.allclose(total, 1e-4)


def test_vn_cleaning_drops_stale_month_and_repairs_spike():
    rows = []
    for i, m in enumerate(pd.period_range("2023-01", "2023-06", freq="M")):
        for tenor in ["1Y", "10Y"]:
            y = 3.0 + 0.01 * i + (0.5 if tenor == "10Y" else 0)
            if str(m) == "2023-03" and tenor == "1Y":
                y = 5.0                                    # spike that reverts
            rows.append({"month": str(m), "tenor": tenor, "yield_pct": y})
    rows += [{"month": "2023-07", "tenor": r["tenor"], "yield_pct": r["yield_pct"]}
             for r in rows if r["month"] == "2023-06"]     # stale copy of June
    h = clean_vn_yields(pd.DataFrame(rows))
    assert pd.Timestamp("2023-07-31") not in h.yields.index
    assert abs(h.yields.loc["2023-03-31", 1.0] - 3.02) < 1e-9
    assert len(h.flags) == 2


# --------------------------------------------------------------------------- pricing
def test_par_bond_prices_near_par_and_yield_matches_coupon():
    c = flat_curve(4.0)
    b = par_bond(SETTLE, 10, c)
    dirty = b.dirty_from_curve(SETTLE, c)
    assert abs(dirty - 100) < 0.1                       # date roll vs exact year fractions
    assert abs(b.ytm(SETTLE, dirty - b.accrued(SETTLE)) - b.coupon_pct) < 0.02


def test_yield_price_roundtrip_and_dv01_sign():
    b = Bond(pd.Timestamp("2033-03-15"), 5.0, freq=1)
    clean = b.clean_from_yield(SETTLE, 4.2)
    assert abs(b.ytm(SETTLE, clean) - 4.2) < 1e-8
    r = b.risk_from_yield(SETTLE, 4.2)
    bump = b.dirty_from_yield(SETTLE, 4.21) - b.dirty_from_yield(SETTLE, 4.2)
    assert r["dv01"] > 0 and abs(-bump - r["dv01"]) / r["dv01"] < 0.01


def test_key_rate_dv01_adds_up_to_parallel():
    c = flat_curve(4.5)
    b = par_bond(SETTLE, 7, c)
    krd = b.key_rate_dv01(SETTLE, c, TENORS)
    parallel = b.dirty_from_curve(SETTLE, c) - b.dirty_from_curve(SETTLE, c.shifted(scenario_shift({0: 1, 40: 1})))
    assert abs(krd.sum() - parallel) < 1e-6


def test_z_spread_recovers_added_spread():
    c = flat_curve(4.0)
    b = Bond(pd.Timestamp("2031-06-30"), 4.5)
    dirty = b.dirty_from_curve(SETTLE, c, spread_bp=35)
    assert abs(b.z_spread(SETTLE, c, dirty - b.accrued(SETTLE)) - 35) < 1e-6


def test_cip_forward_premium_sign():
    vnd, usd = flat_curve(6.0), flat_curve(4.0)
    f = cip_forward(25_000, 1.0, vnd, usd)
    assert abs(f - 25_000 * np.exp(0.02)) < 1e-6        # higher VND rate -> USD at a premium
    mm = cip_forward_money_market(25_000, 365, 6.0, 4.0)
    assert mm > 25_000


def test_garman_kohlhagen_put_call_parity_and_implied_vol():
    S, K, T, rd, rf, v = 26_000, 26_300, 0.5, 5.0, 4.0, 3.0
    c = garman_kohlhagen(S, K, T, rd, rf, v, "call")
    p = garman_kohlhagen(S, K, T, rd, rf, v, "put")
    assert abs((c["price"] - p["price"]) - (S * np.exp(-rf / 100 * T) - K * np.exp(-rd / 100 * T))) < 1e-8
    assert abs(implied_vol(c["price"], S, K, T, rd, rf, "call") - v) < 1e-6
    assert 0 < c["delta"] < 1 and -1 < p["delta"] < 0 and c["gamma"] > 0 and c["vega"] > 0


def test_ewma_vol_scale():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.001, 2000))
    assert abs(ewma_vol(r) - 0.1 * np.sqrt(252)) < 0.6


# --------------------------------------------------------------------------- portfolio
@pytest.fixture(scope="module")
def universe(synthetic_history):
    last = synthetic_history.iloc[-1]
    curve = fit_curve(last.index.values, last.values, 1, "NS")
    return curve, build_universe(curve, SETTLE, TENORS, 1.0), yield_change_cov(synthetic_history, 1.0)


def test_optimiser_respects_constraints(universe):
    _, u, cov = universe
    w = optimise(u, cov, risk_aversion=0.1, max_weight=0.3, max_duration=5.0)
    st = portfolio_stats(w, u, cov)
    assert abs(w.sum() - 1) < 1e-9 and (w >= -1e-12).all() and (w <= 0.3 + 1e-6).all()
    assert st["duration"] <= 5.0 + 1e-6


def test_frontier_is_monotone(universe):
    _, u, cov = universe
    fr = efficient_frontier(u, cov, max_weight=0.4, n_points=8)
    assert len(fr) >= 5
    assert (np.diff(fr["vol_pct"]) > 0).all()
    assert (np.diff(fr["exp_return_pct"]) > -1e-6).all()


def test_stress_parallel_up_loses_and_matches_duration(universe, synthetic_history):
    curve, u, cov = universe
    w = np.full(len(TENORS), 1 / len(TENORS))
    st = stress_test(w, u, curve, standard_scenarios(synthetic_history))
    up = st.set_index("scenario").loc["Parallel +100 bp", "portfolio_pnl_pct"]
    dur = portfolio_stats(w, u, cov)["duration"]
    assert up < 0 and abs(-up - dur) / dur < 0.15     # first-order duration approximation


# --------------------------------------------------------------------------- curve forecasting
def test_dns_factors_reproduce_noise_free_curve():
    L = ns_loadings(TENORS, 0.5)
    Y = pd.DataFrame([L @ np.array([4.0, -1.0, 0.5])], columns=[float(t) for t in TENORS])
    f = dns_factors(Y, 0.5).iloc[0].to_numpy()
    assert np.allclose(f, [4.0, -1.0, 0.5])


def test_curve_backtest_has_no_lookahead(synthetic_history):
    bt = backtest_curve(synthetic_history, horizons=(1,), min_train=24)
    # forecasts made at an origin must not change if later data is altered
    altered = synthetic_history.copy()
    altered.iloc[-1] += 5.0
    bt2 = backtest_curve(altered, horizons=(1,), min_train=24)
    first = bt["origin"].min()
    a = bt[bt["origin"] == first].sort_values(["model", "tenor"])["point"].to_numpy()
    b = bt2[bt2["origin"] == first].sort_values(["model", "tenor"])["point"].to_numpy()
    assert np.allclose(a, b)
    summ = summarize_curve_backtest(bt)
    assert set(summ["model"]) == {"Random walk", "DNS-AR(1)", "AR(1) on yields"}


def test_forecast_intervals_ordered(synthetic_history):
    fc = forecast_curve(synthetic_history, 3)
    assert (fc["lower"] <= fc["point"]).all() and (fc["point"] <= fc["upper"]).all()


def test_continuous_tail_skips_gap():
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2021-01-31", "2021-02-28", "2021-03-31"])
    df = pd.DataFrame({"a": range(5)}, index=idx)
    assert continuous_tail(df).index[0] == pd.Timestamp("2021-01-31")


# --------------------------------------------------------------------------- SQL warehouse
def test_warehouse_queries_run_and_user_sql_is_sandboxed():
    from src.warehouse import build_warehouse, load_queries, run_query, run_user_query
    con = build_warehouse()
    _, queries = load_queries()
    assert len(queries) >= 4
    for q in queries:
        assert len(run_query(con, q.sql)) > 0, q.name
    for bad in ["select * from read_csv('/etc/passwd')", "drop table vn_yields", "select 1; select 2"]:
        with pytest.raises(Exception):
            run_user_query(con, bad)
    assert run_user_query(con, "select count(*) as n from vn_yields")["n"].iloc[0] > 0
