"""Page: VND government bond portfolio – carry and roll-down vs curve risk, optimisation, stress tests."""
import numpy as np
import pandas as pd
import streamlit as st

from src.app_common import footer, vn_curve, vn_history
from src.portfolio import (build_universe, efficient_frontier, optimise, portfolio_stats,
                           standard_scenarios, stress_test, yield_change_cov)
from src.rates_plotting import AMBER, BLUE, bar_figure, frontier_figure

st.title("VND government bond portfolio")
st.markdown(
    "How much carry and roll-down each tenor offers on today's curve, how much curve risk it adds, "
    "and the best mix under a desk's duration and concentration limits."
)

yields, _, _, ts = vn_history()
asof = yields.index[-1]
cv = vn_curve(asof)
tenors = list(yields.columns)

with st.sidebar:
    st.header("Portfolio limits")
    horizon_m = st.select_slider("Horizon", [3, 6, 12], value=12, format_func=lambda m: f"{m} months")
    max_w = st.slider("Max weight per tenor", 0.15, 1.0, 0.40, 0.05, format="%.2f")
    use_dur = st.checkbox("Duration limit", value=True)
    max_dur = st.slider("Max modified duration", 1.0, 15.0, 6.0, 0.5) if use_dur else None
    risk_aversion = st.select_slider("Risk aversion λ", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1,
                                     help="Objective: carry+roll − (λ/2)·variance, both in % over the horizon.")

H = horizon_m / 12


@st.cache_resource(show_spinner="Pricing the universe…")
def universe(asof_key: str, horizon: float):
    return build_universe(cv, asof, tenors, horizon)


u = universe(f"{asof:%Y-%m}", H)
cov = yield_change_cov(ts, H)

try:
    w = optimise(u, cov, risk_aversion=risk_aversion, max_weight=max_w, max_duration=max_dur)
    frontier = efficient_frontier(u, cov, max_weight=max_w, max_duration=max_dur, n_points=25)
except ValueError as e:
    st.error(str(e))
    st.stop()
stats = portfolio_stats(w, u, cov)

c = st.columns(4)
c[0].metric(f"Carry + roll-down ({horizon_m}m)", f"{stats['exp_return_pct']:.2f}%")
c[1].metric("Risk (P&L vol, horizon)", f"{stats['vol_pct']:.2f}%")
c[2].metric("Modified duration", f"{stats['duration']:.2f}")
c[3].metric("DV01 per 100 bn VND", f"{stats['dv01_per_100'] * 1e9:,.0f} VND/bp")

tab_opt, tab_univ, tab_stress, tab_method = st.tabs(
    ["⚖️ Optimal mix", "📋 Universe", "🌪️ Stress tests", "📘 Methodology"])

bond_vols = pd.Series([portfolio_stats(np.eye(len(tenors))[i], u, cov)["vol_pct"] for i in range(len(tenors))],
                      index=u.table.index)

with tab_opt:
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(frontier_figure(frontier, u.table, stats, bond_vols), width="stretch")
    with right:
        st.plotly_chart(bar_figure([f"{t:g}Y" for t in tenors], w * 100, "Weight (%)", AMBER, ".1f", 400),
                        width="stretch")
    top = frontier["exp_return_pct"].max()
    knee = frontier[frontier["exp_return_pct"] >= top - 0.02].iloc[0]
    st.markdown(
        f"- The frontier flattens: beyond about **{knee['vol_pct']:.1f}% risk** (duration {knee['duration']:.1f}) "
        f"extra duration adds less than 2 bp of carry+roll while risk keeps rising. On this curve the long end "
        f"is not paid for its extra volatility.\n"
        f"- Selected mix: {', '.join(f'{t:g}Y {x:.0%}' for t, x in zip(tenors, w) if x > 0.005)}.\n"
        f"- Expected return assumes the curve stays where it is; it is the neutral scenario a desk compares "
        f"against, not a forecast (the Rates page shows that forecasting yields is hard)."
    )

with tab_univ:
    t = u.table.copy()
    t.index = [f"{x:g}Y" for x in t.index]
    t["risk_pct"] = bond_vols.to_numpy()
    st.dataframe(t.round(3), width="stretch")
    st.caption(f"Par bonds on the VND curve of {asof:%d %b %Y}; carry = coupon over {horizon_m} months, roll-down = "
               "price change from ageing on an unchanged curve. Risk uses the covariance of monthly yield changes "
               f"{ts.index[0]:%b %Y}–{ts.index[-1]:%b %Y}, scaled to the horizon.")
    vol_bp = pd.Series(np.sqrt(np.diag(cov)), index=[f"{x:g}Y" for x in cov.columns])
    st.plotly_chart(bar_figure(vol_bp.index, vol_bp.to_numpy(), f"Yield change std over {horizon_m}m (bp)",
                               BLUE, ".0f", 280), width="stretch")

with tab_stress:
    st_tbl = stress_test(w, u, cv, standard_scenarios(ts))
    eq = stress_test(np.full(len(tenors), 1 / len(tenors)), u, cv, standard_scenarios(ts))
    show = st_tbl[["scenario", "portfolio_pnl_pct"]].rename(columns={"portfolio_pnl_pct": "Selected mix (%)"})
    show["Equal weights (%)"] = eq["portfolio_pnl_pct"].to_numpy()
    show["10Y bond alone (%)"] = st_tbl["10Y"].to_numpy()
    st.dataframe(show.round(2), hide_index=True, width="stretch")
    worst = st_tbl.loc[st_tbl["portfolio_pnl_pct"].idxmin()]
    st.markdown(f"Instant full revaluation. Worst case for the selected mix: **{worst['scenario']}**, "
                f"**{worst['portfolio_pnl_pct']:.1f}%** of market value, roughly "
                f"{-worst['portfolio_pnl_pct'] / max(stats['exp_return_pct'], 1e-9):.1f}× its expected "
                f"{horizon_m}-month carry+roll.")
    with st.expander("P&L by tenor under each scenario (%)"):
        st.dataframe(st_tbl.round(2), hide_index=True, width="stretch")

with tab_method:
    st.markdown(r"""
#### Expected return: carry and roll-down
For each tenor a par bond is priced on the calibrated zero curve. Its value at the horizon, with the curve
unchanged, is the remaining cash flows discounted on the same curve plus coupons received and reinvested.
The gain over today's price splits into coupon (carry) and price change from moving down the curve (roll-down).

#### Risk
Key-rate DV01s map each bond's P&L to moves at the eight published tenors. With $K$ the matrix of key-rate
DV01s (% of value per bp) and $\Sigma$ the covariance of yield changes over the horizon (bp²),
$\sigma_p^2 = w^\top K \Sigma K^\top w$. Monthly changes are assumed independent, so $\Sigma$ is the monthly
covariance × number of months.

#### Optimisation
SLSQP (SciPy): maximise $w^\top\mu - \tfrac{\lambda}{2}\sigma_p^2$ subject to $\sum w = 1$, $0 \le w \le w_{max}$
and $w^\top D \le D_{max}$. The frontier maximises $w^\top\mu$ for a grid of risk levels.

#### Limitations
- Generic par bonds, not actual bond lines; no liquidity, bid-ask or repo funding cost.
- Covariance from 68 months, dominated by the 2022 rate rise; i.i.d. scaling ignores trends.
- Static-curve carry ignores expected curve moves by construction.
""")

footer()
