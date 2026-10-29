"""Interbank liquidity tab of the Rates & liquidity page (imported by views/rates_liquidity.py)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.interbank import (STRESS_BP, STRESS_WINDOW, TENORS, backtest_overnight, backtest_stress, brier_skill,
                           calendar_effects, fit_stress_model, load_interbank, summarize_overnight, wide)
from src.rates_plotting import AMBER, BLUE, GREY, INK, TEAL, _layout

SBV_ON_LENDING = (pd.Timestamp("2023-06-19"), 5.0)      # Decision 1123/QĐ-NHNN of 16 June 2023


@st.cache_data(show_spinner="Loading SBV interbank data…")
def _data():
    df = load_interbank()
    rates, vols = wide(df, "rate_pct"), wide(df, "volume_bn")
    on = rates["ON"].dropna()
    on = on.reindex(pd.bdate_range(on.index.min(), on.index.max())).ffill()   # holidays: last value
    return df, rates, vols, on


@st.cache_data(show_spinner="Backtesting overnight-rate and stress models…")
def _models(key: str, _on: pd.Series):
    bt = backtest_overnight(_on)
    bs = backtest_stress(_on)
    return bt, summarize_overnight(bt), bs, brier_skill(bs), calendar_effects(_on)


def render_liquidity() -> None:
    df, rates, vols, on = _data()
    last = rates.index.max()
    total = vols.sum(axis=1)
    c = st.columns(4)
    c[0].metric(f"Overnight rate ({last:%d %b %Y})", f"{rates['ON'].dropna().iloc[-1]:.2f}%",
                f"{(rates['ON'].dropna().iloc[-1] - rates['ON'].dropna().iloc[-6]) * 100:+.0f} bp w/w", delta_color="off")
    c[1].metric("3M interbank", f"{rates['3M'].dropna().iloc[-1]:.2f}%")
    c[2].metric("Turnover, all tenors", f"{total.iloc[-1] / 1000:,.0f} tn VND")
    c[3].metric("Overnight share of turnover (20d)",
                f"{(vols['ON'] / total).iloc[-20:].mean():.0%}")

    lookback = st.select_slider("History", ["1 year", "3 years", "5 years", "All"], value="3 years", key="liq_lb")
    start = last - pd.DateOffset(years={"1 year": 1, "3 years": 3, "5 years": 5, "All": 20}[lookback])
    r = rates[rates.index >= start]
    fig = go.Figure()
    for t, col in zip(["ON", "1M", "3M"], (BLUE, TEAL, AMBER)):
        fig.add_trace(go.Scatter(x=r.index, y=r[t], name=t, mode="lines", line=dict(color=col, width=1.6),
                                 hovertemplate="%{y:.2f}%"))
    d0, lvl = SBV_ON_LENDING
    if r.index.max() >= d0:
        fig.add_trace(go.Scatter(x=[max(d0, r.index.min()), r.index.max()], y=[lvl, lvl],
                                 name="SBV overnight lending rate 5.0%", mode="lines",
                                 line=dict(color=GREY, dash="dash", width=1.5), hoverinfo="skip"))
    fig = _layout(fig, "Rate (% a year)", 360)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("State Bank of Vietnam daily averages. Days on which a tenor did not trade (SBV repeats an "
               "earlier figure) are left out. The dashed line is SBV's overnight lending rate in interbank "
               "electronic payments (5.0% since 19 June 2023, Decision 1123/QĐ-NHNN).")

    bt, summ, bs, skill, cal = _models(f"{last:%Y-%m-%d}|{len(on)}", on)

    st.markdown("##### Calendar effects on the overnight rate")
    show = cal.copy()
    show["window"] = show["window"].map({"month_end": "Last 3 business days of a month (non-quarter)",
                                         "quarter_end": "Last 3 business days of a quarter",
                                         "pre_tet": "10 business days before Tết"})
    st.dataframe(show.round({"effect_bp": 1, "t_stat": 2, "p_value": 3}), hide_index=True, width="stretch")
    st.caption("Effect = average overnight rate in the window minus the same month's average, in bp; "
               "OLS with Newey–West (5 lags) standard errors.")

    st.markdown("##### Forecasting the overnight rate")
    st.dataframe(summ.round(3), hide_index=True, width="stretch",
                 column_config={"h": "Horizon (business days)", "RW_RMSE_bp": "Random walk RMSE (bp)",
                                "HAR_RMSE_bp": "HAR + calendar RMSE (bp)", "rel_RMSE": "HAR ÷ RW",
                                "DM_pvalue": "DM p-value"})
    st.caption(f"Expanding window from {on.index[0]:%b %Y}, refit every 5 days, first forecast after 3 years; "
               f"{bt['origin'].min():%b %Y} → {bt['origin'].max():%b %Y}. HAR regressors: last day, 5-day and "
               "22-day averages, plus month-end, quarter-end and pre-Tết dummies for the target day.")

    st.markdown(f"##### Liquidity-stress warning (overnight +{STRESS_BP} bp within {STRESS_WINDOW} days)")
    _, _, _, p_now, coefs, d_now = fit_stress_model(on)
    k = st.columns(4)
    k[0].metric(f"Probability now ({d_now:%d %b %Y})", "already in stress" if np.isnan(p_now) else f"{p_now:.0%}")
    k[1].metric("Historical base rate", f"{skill['event_rate']:.0%}")
    k[2].metric("Brier skill vs base rate", f"{skill['skill']:+.2f}",
                help="1 − Brier(model) / Brier(base rate), out of sample. Above 0 = more informative than the base rate.")
    k[3].metric("Out-of-sample days", f"{skill['n']:,}")
    by = bs.assign(bucket=pd.cut(bs["p_model"], [0, .1, .2, .4, 1.0], include_lowest=True))
    calib = by.groupby("bucket", observed=True).agg(days=("y", "size"), predicted=("p_model", "mean"),
                                                   observed=("y", "mean")).reset_index()
    calib["bucket"] = calib["bucket"].astype(str)
    c1, c2 = st.columns([3, 2])
    with c1:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"),
                                name="Perfect calibration", hoverinfo="skip"))
        f2.add_trace(go.Scatter(x=calib["predicted"], y=calib["observed"], mode="markers+lines",
                                marker=dict(size=10, color=BLUE, line=dict(color="white", width=2)),
                                line=dict(color=BLUE, width=2), name="Model (out of sample)",
                                customdata=calib["days"], hovertemplate="predicted %{x:.0%}, observed %{y:.0%}, %{customdata} days"))
        f2 = _layout(f2, "Observed frequency", 320, "Predicted probability", ".0%")
        f2.update_xaxes(tickformat=".0%", range=[0, max(0.6, calib["predicted"].max() + 0.05)])
        st.plotly_chart(f2, width="stretch")
    with c2:
        st.dataframe(calib.round(3), hide_index=True, width="stretch")
        st.caption(f"Only days on which the overnight rate is less than {STRESS_BP} bp above its 20-day median "
                   "count, so the model predicts the start of a new episode, not its continuation. "
                   "Logistic regression coefficients (standardised): "
                   + ", ".join(f"{n} {v:+.2f}" for n, v in coefs.items()))
