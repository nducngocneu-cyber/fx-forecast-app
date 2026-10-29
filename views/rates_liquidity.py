"""Page: VND government yield-curve forecasting and interbank liquidity."""
import numpy as np
import pandas as pd
import streamlit as st

from src.app_common import footer, vn_history
from src.rates_forecast import backtest_curve, forecast_curve, summarize_curve_backtest
from src.rates_plotting import (curve_forecast_figure, rel_rmse_by_tenor_figure,
                                yield_history_figure)

st.title("VND rates & liquidity")
st.markdown(
    "Monthly Vietnamese government bond yields (VBMA) forecast with a dynamic Nelson–Siegel "
    "model and benchmarked against the random walk, plus interbank liquidity from the State Bank "
    "of Vietnam."
)

yields, raw, flags, ts = vn_history()


@st.cache_data(show_spinner="Running the expanding-window backtest…")
def curve_backtest(key: str, _ts: pd.DataFrame):
    bt = backtest_curve(_ts, horizons=(1, 3, 6), min_train=24)
    return bt, summarize_curve_backtest(bt)


tab_curve, tab_bt, tab_liq, tab_data, tab_method = st.tabs(
    ["📉 Curve forecast", "🧪 Backtest", "💧 Interbank liquidity", "🗂️ Data", "📘 Methodology"])

# --------------------------------------------------------------------------- forecast
with tab_curve:
    last = ts.iloc[-1].rename(ts.index[-1])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"10Y yield ({ts.index[-1]:%b %Y})", f"{last[10.0]:.2f}%",
              f"{(last[10.0] - ts[10.0].iloc[-13]) * 100:+.0f} bp y/y", delta_color="off")
    c2.metric("2Y yield", f"{last[2.0]:.2f}%",
              f"{(last[2.0] - ts[2.0].iloc[-13]) * 100:+.0f} bp y/y", delta_color="off")
    c3.metric("2s10s slope", f"{(last[10.0] - last[2.0]) * 100:.0f} bp")
    c4.metric("Months of clean history", f"{len(ts)}")

    st.plotly_chart(yield_history_figure(yields), width="stretch")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        h = st.radio("Horizon", [1, 3, 6], index=1, format_func=lambda x: f"{x} month{'s' * (x > 1)}")
        model = st.radio("Model", ["DNS-AR(1)", "AR(1) on yields", "Random walk"])
        level = st.select_slider("Band", [0.8, 0.9], value=0.9, format_func=lambda x: f"{x:.0%}")
    fc = forecast_curve(ts, h, level=level)
    with col_b:
        st.plotly_chart(curve_forecast_figure(last, fc, model, h), width="stretch")
    tbl = fc.pivot_table(index="tenor", columns="model", values="point")[["Random walk", "DNS-AR(1)", "AR(1) on yields"]]
    tbl.index = [f"{t:g}Y" for t in tbl.index]
    st.caption(f"Point forecasts (%), λ = {fc.attrs['lambda']:.2f} chosen on the full window. "
               "Check the Backtest tab before reading anything into a predicted move.")
    st.dataframe(tbl.round(2).T, width="stretch")

# --------------------------------------------------------------------------- backtest
with tab_bt:
    bt, summ = curve_backtest(f"{ts.index[-1]:%Y-%m}|{len(ts)}", ts)
    st.markdown(
        f"Expanding window from {ts.index[0]:%b %Y}: at each origin (first after 24 months) every "
        f"model, including the choice of λ, is re-estimated on past data only. "
        f"{bt['origin'].nunique()} origins, {bt['origin'].min():%b %Y} → {bt['origin'].max():%b %Y}."
    )
    hb = st.radio("Horizon", [1, 3, 6], horizontal=True, key="bt_h",
                  format_func=lambda x: f"{x} month{'s' * (x > 1)}")
    st.plotly_chart(rel_rmse_by_tenor_figure(summ, hb), width="stretch")
    st.caption("Diamonds: difference from the random walk significant at 5% (Diebold–Mariano, "
               "Harvey–Leybourne–Newbold correction).")
    s = summ[summ["h"] == hb].copy()
    s["tenor"] = [f"{t:g}Y" for t in s["tenor"]]
    wide = s.pivot_table(index="tenor", columns="model", values=["RMSE_bp", "rel_RMSE", "DM_pvalue", "coverage"],
                         sort=False)
    st.dataframe(wide.round(3), width="stretch")

    st.markdown("##### Takeaways")
    lines = []
    for m in ["DNS-AR(1)", "AR(1) on yields"]:
        g = summ[summ["model"] == m]
        better = g[(g["rel_RMSE"] < 1) & (g["DM_pvalue"] < 0.05)]
        worse = g[(g["rel_RMSE"] > 1) & (g["DM_pvalue"] < 0.05)]
        lines.append(f"- **{m}**: RMSE between {g['rel_RMSE'].min():.2f}× and {g['rel_RMSE'].max():.2f}× the "
                     f"random walk across tenors and horizons; significantly better in {len(better)} and "
                     f"significantly worse in {len(worse)} of {len(g)} tenor–horizon cells.")
    rw = summ[summ["model"] == "Random walk"].groupby("h")["RMSE_bp"].mean()
    lines.append("- Random-walk error averages " + ", ".join(f"{v:.0f} bp at {k} m" for k, v in rw.items())
                 + ". Both mean-reverting models pull yields towards their sample average; in a period "
                 "with a clear trend (falling in 2023, rising in 2025–26) that costs accuracy.")
    lines.append("- About 40 overlapping monthly origins give the tests little power; this is evidence "
                 "about this sample, not a general ranking.")
    st.markdown("\n".join(lines))

# --------------------------------------------------------------------------- liquidity
with tab_liq:
    try:
        from views._liquidity_section import render_liquidity
        render_liquidity()
    except (FileNotFoundError, ImportError):
        st.info("Interbank data snapshot not found. Run `python -m scripts.fetch_interbank` to download "
                "the State Bank of Vietnam's daily interbank rates and volumes.")

# --------------------------------------------------------------------------- data
with tab_data:
    st.markdown(
        "Source: Vietnam Bond Market Association (VBMA), monthly reference yields of Vietnamese "
        "government bonds by tenor, downloaded from its public secondary-market page. Two defects are "
        "repaired before modelling; the table lists every change."
    )
    st.dataframe(flags, hide_index=True, width="stretch")
    gaps = sorted(set(pd.period_range(yields.index.min(), yields.index.max(), freq="M").strftime("%Y-%m"))
                  - set(yields.index.strftime("%Y-%m")))
    st.markdown(f"Months not published by VBMA: {', '.join(gaps) if gaps else 'none'}. Models use the "
                f"continuous run from **{ts.index[0]:%b %Y}** to **{ts.index[-1]:%b %Y}** ({len(ts)} months).")
    show = yields.copy()
    show.columns = [f"{c:g}Y" for c in show.columns]
    st.dataframe(show.sort_index(ascending=False).round(3), width="stretch", height=300)
    st.download_button("Download cleaned yields (CSV)", show.to_csv().encode(), "vn_govt_yields_clean.csv",
                       "text/csv")

# --------------------------------------------------------------------------- method
with tab_method:
    st.markdown(r"""
#### Dynamic Nelson–Siegel (Diebold and Li, 2006)
Each month's curve is summarised by three factors, estimated by OLS with λ fixed:

$$y_t(\tau) = \beta_{1t} + \beta_{2t}\frac{1-e^{-\lambda\tau}}{\lambda\tau} + \beta_{3t}\left(\frac{1-e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right)$$

β₁ is the level, β₂ the slope (short minus long) and β₃ the curvature. λ is chosen by grid search to
minimise fitting error on the training window only. Forecasts use a direct h-step AR(1) per factor,
β̂_{t+h} = c + φ·β_t, as in Diebold and Li (2006).

#### Benchmarks
- **Random walk**: no change. Hard to beat at short horizons for yields as for exchange rates.
- **AR(1) on yields**: the same direct regression tenor by tenor, without the curve structure.

#### Bands
Empirical 5th/95th (or 10th/90th) percentiles of the in-sample h-step errors, so they keep the fat
tails seen in 2022 (the 10Y yield rose by about 280 bp in twelve months).

#### Limitations
- Monthly data from 2021 only: about 68 observations, 40 forecast origins.
- VBMA yields are reference levels of coupon bonds, not zero-coupon yields; the factor model is fitted to
  them directly, as is common in practice. The pricing page calibrates a proper zero curve.
- No macro information (policy rates, inflation, interbank rates) enters the forecasts.

#### References
- Diebold, F.X. and Li, C. (2006) 'Forecasting the term structure of government bond yields', *Journal of Econometrics*, 130(2), pp. 337–364.
- Nelson, C.R. and Siegel, A.F. (1987) 'Parsimonious modeling of yield curves', *Journal of Business*, 60(4), pp. 473–489.
- Diebold, F.X. and Mariano, R.S. (1995) 'Comparing predictive accuracy', *Journal of Business & Economic Statistics*, 13(3), pp. 253–263.
""")

footer()
