"""Page: USD/VND short-term forecast (random walk, ARIMA, gradient boosting + backtest)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src.backtest import rolling_backtest, summarize
from src.config import (DATA_DIR, DEFAULT_TRAIN_WINDOW, MAX_HORIZON, TRADING_DAYS_PER_YEAR,
                        TRAIN_WINDOWS)
from src.data_loader import FXData, load_fx_data
from src.models import MODEL_REGISTRY, BaseForecaster
from src.plotting import (coverage_figure, forecast_figure, history_figure,
                          rmse_by_horizon_figure)

from src.app_common import footer


SOURCE_OPTIONS = {
    "Live (Yahoo Finance), fallback to snapshot": "auto",
    "Bundled snapshot (offline)": "snapshot",
    "Simulated demo data": "simulated",
}


# --------------------------------------------------------------------------- #
# Cached computations
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=6 * 3600, show_spinner="Loading USD/VND data…")
def get_data(mode: str) -> FXData:
    return load_fx_data(mode)


def data_key(d: FXData) -> str:
    s = d.series
    return f"{d.source}|{s.index[-1]:%Y-%m-%d}|{len(s)}|{s.iloc[-1]:.2f}"


@st.cache_resource(show_spinner=False, max_entries=24)
def fitted_model(key: str, model_name: str, train_size: int, _series: pd.Series) -> BaseForecaster:
    """Fit once per (data version, model, window); ``_series`` is excluded from hashing."""
    return MODEL_REGISTRY[model_name]().fit(_series.iloc[-train_size:])


@st.cache_data(show_spinner=False)
def precomputed_backtest():
    path = DATA_DIR / "backtest_results.csv"
    if not path.exists():
        return None, None
    bt = pd.read_csv(path, parse_dates=["origin"])
    meta = json.loads((DATA_DIR / "backtest_meta.json").read_text())
    return bt, meta


@st.cache_data(show_spinner=False, max_entries=8)
def live_backtest(key: str, train_size: int, _series: pd.Series):
    settings = dict(train_size=train_size, horizon=30, n_origins=40, step=7, level=0.90)
    bt = rolling_backtest(_series, **settings)
    meta = {**settings, "data_end": key.split("|")[1],
            "first_origin": f"{bt['origin'].min():%Y-%m-%d}",
            "last_origin": f"{bt['origin'].max():%Y-%m-%d}"}
    return bt, meta


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    source_label = st.radio("Data source", list(SOURCE_OPTIONS), index=0)
    horizon = st.slider("Forecast horizon (business days)", 1, MAX_HORIZON, 14,
                        help="Business days ahead. 5 ≈ one week, 21 ≈ one month.")
    models_selected = st.multiselect("Models", list(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    level = st.select_slider("Prediction interval", options=[0.80, 0.90, 0.95], value=0.90,
                             format_func=lambda x: f"{x:.0%}")
    train_label = st.selectbox("Training window", list(TRAIN_WINDOWS),
                               index=list(TRAIN_WINDOWS).index(DEFAULT_TRAIN_WINDOW))
    lookback_label = st.selectbox("Chart history", ["3 months", "6 months", "1 year", "3 years"], index=1)
    st.divider()
    st.caption("Portfolio project – not investment advice. Forecasts are statistical "
               "estimates with wide uncertainty.")

train_size = TRAIN_WINDOWS[train_label]
lookback = {"3 months": 63, "6 months": 126, "1 year": 252, "3 years": 756}[lookback_label]

# --------------------------------------------------------------------------- #
# Data + header
# --------------------------------------------------------------------------- #
data = get_data(SOURCE_OPTIONS[source_label])
y = data.series
key = data_key(data)

st.title("USD/VND short-term forecast")
st.markdown(
    "Daily USD/VND forecasts from a random-walk benchmark, ARIMA and gradient boosting, "
    "evaluated out-of-sample with a walk-forward backtest."
)

if data.is_simulated:
    st.warning(f"⚠️ **{data.label}.** Numbers below are illustrative and do not describe the market.")
else:
    st.info(f"**Data:** {data.label}. {data.n_cleaned} bad ticks removed by the spike filter "
            "(see *Data* tab).", icon="ℹ️")

spot = float(y.iloc[-1])
log_ret = np.log(y).diff().dropna()


def pct_change(days: int) -> float:
    return (spot / float(y.iloc[-days - 1]) - 1) * 100 if len(y) > days else np.nan


c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Last close ({y.index[-1]:%d %b %Y})", f"{spot:,.0f}")
c2.metric("1-month change", f"{pct_change(21):+.2f}%")
c3.metric("1-year change", f"{pct_change(252):+.2f}%")
c4.metric("Realised vol (20d, ann.)",
          f"{log_ret.iloc[-20:].std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100:.2f}%")

tab_fc, tab_bt, tab_data, tab_method = st.tabs(["📈 Forecast", "🧪 Backtest", "🗂️ Data", "📘 Methodology"])

# --------------------------------------------------------------------------- #
# Forecast tab
# --------------------------------------------------------------------------- #
with tab_fc:
    if not models_selected:
        st.warning("Select at least one model in the sidebar.")
    else:
        forecasts, specs = {}, {}
        with st.spinner("Fitting models…"):
            for name in models_selected:
                m = fitted_model(key, name, train_size, y)
                forecasts[name] = m.predict(horizon, level)
                specs[name] = m.describe()

        st.plotly_chart(forecast_figure(y, forecasts, lookback, level), width="stretch")

        end_date = next(iter(forecasts.values())).index[-1]
        rows = []
        for name, fc in forecasts.items():
            last = fc.iloc[-1]
            rows.append({
                "Model": name,
                "Point": last["point"],
                "Lower": last["lower"],
                "Upper": last["upper"],
                "Change vs spot (%)": (last["point"] / spot - 1) * 100,
                "Band half-width (%)": (last["upper"] - last["lower"]) / 2 / spot * 100,
                "Specification": specs[name],
            })
        table = pd.DataFrame(rows).round({"Point": 0, "Lower": 0, "Upper": 0})
        st.subheader(f"Forecast for {end_date:%d %b %Y} (t + {horizon} business days)")
        st.dataframe(
            table, hide_index=True,
            column_config={
                "Point": st.column_config.NumberColumn(format="localized"),
                "Lower": st.column_config.NumberColumn(f"Lower ({level:.0%})", format="localized"),
                "Upper": st.column_config.NumberColumn(f"Upper ({level:.0%})", format="localized"),
                "Change vs spot (%)": st.column_config.NumberColumn(format="%+.2f"),
                "Band half-width (%)": st.column_config.NumberColumn(format="±%.2f"),
            },
        )

        # Plain-language reading, computed from the numbers on screen
        max_move = table["Change vs spot (%)"].abs().max()
        min_band = table["Band half-width (%)"].min()
        st.markdown("##### How to read this")
        st.markdown(
            f"- The largest predicted move across models is **{max_move:.2f}%**, while the "
            f"narrowest {level:.0%} band is **±{min_band:.2f}%**. "
            + ("The predicted move is small relative to the uncertainty, so the band carries most of the information."
               if max_move < min_band / 2 else
               "At least one model predicts a move that is large relative to its band; check the backtest before relying on it.")
            + "\n- Point forecasts are medians; bands widen with the horizon because daily shocks accumulate."
            "\n- The *Backtest* tab shows whether each model has beaten the random walk out-of-sample."
        )

        out = pd.concat({k: v for k, v in forecasts.items()}, axis=1)
        out.columns = [f"{m}_{c}" for m, c in out.columns]
        st.download_button("Download forecast (CSV)", out.round(2).to_csv().encode(),
                           file_name=f"usdvnd_forecast_{horizon}d.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Backtest tab
# --------------------------------------------------------------------------- #
with tab_bt:
    st.markdown(
        "Walk-forward evaluation: at each origin the models are re-estimated on data available "
        "up to that day only, then forecast 1–30 business days ahead. Accuracy is compared with "
        "the random walk, the standard benchmark for exchange rates."
    )
    # Precomputed results describe the bundled real-data snapshot, so they are shown
    # for real data only. Simulated data always gets its own backtest.
    bt, meta, origin_label = None, None, ""
    custom = st.session_state.get("bt_custom")
    rerun = st.button(f"Re-run on current data with a {train_label} window (≈15–60 s)")
    if rerun:
        with st.spinner("Running walk-forward backtest…"):
            bt, meta = live_backtest(key, train_size, y)
        st.session_state["bt_custom"] = (key, train_size, bt, meta)
        origin_label = "current data"
    elif custom and custom[0] == key:
        _, _, bt, meta = custom
        origin_label = "current data"
    elif data.is_simulated:
        st.info("The precomputed backtest uses real data. Click the button above to backtest "
                "the simulated series.")
    else:
        bt, meta = precomputed_backtest()
        origin_label = "bundled snapshot (precomputed)"

    if bt is None:
        if not data.is_simulated:
            st.info("No precomputed backtest found. Click the button above to run one.")
    else:
        summ = summarize(bt, step=meta["step"])
        st.caption(
            f"Computed on **{origin_label}**{' – simulated series' if data.is_simulated else ''}: "
            f"{meta['n_origins']} origins every {meta['step']} business days, "
            f"{meta['first_origin']} → {meta['last_origin']}; training window "
            f"{meta['train_size']} days; {meta['level']:.0%} intervals; data to {meta['data_end']}."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Relative RMSE (below 1 = beats random walk)**")
            st.plotly_chart(rmse_by_horizon_figure(summ), width="stretch")
        with col_b:
            st.markdown(f"**Interval coverage (target {meta['level']:.0%})**")
            st.plotly_chart(coverage_figure(summ, meta["level"]), width="stretch")

        show_h = [1, 5, 10, 14, 20, 30]
        tbl = summ[summ["h"].isin(show_h)][
            ["h", "model", "RMSE", "MAE", "rel_RMSE", "DM_pvalue", "coverage", "hit_rate"]
        ].sort_values(["h", "model"])
        st.dataframe(
            tbl, hide_index=True,
            column_config={
                "h": "Horizon (d)",
                "RMSE": st.column_config.NumberColumn("RMSE (VND)", format="%.1f"),
                "MAE": st.column_config.NumberColumn("MAE (VND)", format="%.1f"),
                "rel_RMSE": st.column_config.NumberColumn("RMSE ÷ RW", format="%.3f"),
                "DM_pvalue": st.column_config.NumberColumn("DM p-value vs RW", format="%.3f"),
                "coverage": st.column_config.NumberColumn("Coverage", format="%.2f"),
                "hit_rate": st.column_config.NumberColumn("Direction hit rate", format="%.2f"),
            },
        )

        # Takeaways generated from the table
        st.markdown("##### Takeaways")
        lines = []
        for name in [m for m in MODEL_REGISTRY if m != "Random walk"]:
            g = summ[summ["model"] == name]
            beats = g[(g["rel_RMSE"] < 1) & (g["DM_pvalue"] < 0.05)]["h"].tolist()
            worse = g[(g["rel_RMSE"] > 1) & (g["DM_pvalue"] < 0.05)]["h"].tolist()
            lo, hi = g["rel_RMSE"].min(), g["rel_RMSE"].max()
            msg = f"- **{name}**: RMSE between {lo:.2f}× and {hi:.2f}× the random walk across horizons. "
            if beats:
                msg += f"Significantly better at 5% for h = {', '.join(map(str, beats))}. "
            elif worse:
                msg += f"Significantly worse at 5% for h = {', '.join(map(str, worse))}; not significantly better at any horizon. "
            else:
                msg += "No horizon where the difference is statistically significant at 5%. "
            lines.append(msg)
        cov = summ.groupby("model")["coverage"].mean()
        lines.append("- Average interval coverage: " + ", ".join(f"{m} {c:.0%}" for m, c in cov.items())
                     + f" (target {meta['level']:.0%}). Coverage well above target means bands are wider than needed.")
        lines.append(f"- With {meta['n_origins']} overlapping origins the tests have limited power; "
                     "treat differences of a few percent as noise.")
        st.markdown("\n".join(lines))

# --------------------------------------------------------------------------- #
# Data tab
# --------------------------------------------------------------------------- #
with tab_data:
    st.markdown(
        "Free FX feeds for VND contain isolated one-day spikes that revert the next day. "
        "A quote is replaced by the previous valid value when it deviates more than 0.6% from "
        "its centred 11-day median; persistent level shifts are kept."
    )
    st.plotly_chart(history_figure(data.raw, y), width="stretch")
    stats_tbl = pd.DataFrame({
        "Statistic": ["Observations (business days)", "First date", "Last date", "Quotes cleaned",
                      "Daily log-return std (bp)", "Share of zero-change days"],
        "Value": [f"{len(y):,}", f"{y.index[0]:%Y-%m-%d}", f"{y.index[-1]:%Y-%m-%d}", f"{data.n_cleaned}",
                  f"{log_ret.std() * 1e4:.1f}", f"{(log_ret == 0).mean():.1%}"],
    })
    st.dataframe(stats_tbl, hide_index=True)
    st.download_button("Download cleaned series (CSV)", y.round(2).to_csv().encode(),
                       file_name="usdvnd_cleaned.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Methodology tab
# --------------------------------------------------------------------------- #
with tab_method:
    st.markdown(r"""
#### Problem
Treasury desks need a short-term view of USD/VND for FX position management, hedging
timing and liquidity planning for client USD demand. What matters is less the point
forecast than a **calibrated range** and an honest comparison with a naive benchmark.

#### Models
| Model | What it does | Interval |
|---|---|---|
| **Random walk** | Tomorrow's rate = today's rate. The benchmark that structural FX models have struggled to beat out-of-sample since Meese & Rogoff (1983). | i.i.d. normal log-returns, σ from the last 250 days, widening with √h |
| **ARIMA(p,1,q)** | Short-run autocorrelation in daily changes, optional drift. Order chosen by AIC (p, q ≤ 2) on the training window. | Analytical, from the state-space model |
| **Gradient boosting** | Non-linear model of next-day log-return on lagged returns, 5/20-day momentum and volatility, distance to the 20-day average. Multi-step forecasts are recursive. | 500 simulated paths with residuals bootstrapped from a hold-out split |

All models are estimated on **log prices**, so bands are asymmetric in VND and point
forecasts are medians.

#### Evaluation
Walk-forward backtest: re-estimate at each origin using past data only (ARIMA order fixed
from the first window to avoid look-ahead), forecast 1–30 days, compare with realised values.
Metrics: RMSE/MAE in VND, RMSE relative to the random walk, Diebold–Mariano test with the
Harvey–Leybourne–Newbold correction, interval coverage and direction hit rate.

#### Why not Prophet?
Prophet is designed for series with strong calendar seasonality (retail demand, web
traffic). Daily FX has little such seasonality, and the package adds a heavy build to a
free cloud deployment. ARIMA plus a tree model covers the classical and ML sides at lower cost.

#### Limitations
- Yahoo's `VND=X` is an aggregated indicative quote. It can differ from bank interbank
  prices and from the State Bank of Vietnam (SBV) central rate.
- USD/VND trades inside a band around the SBV central rate (±5% since 17 Oct 2022). The models
  do not use the central rate, policy announcements, interest-rate differentials or the DXY.
- Bootstrapped residuals ignore volatility clustering; bands may be too narrow in stress periods.
- Vietnamese public holidays are forward-filled on a Mon–Fri calendar.

#### References
- Diebold, F.X. and Mariano, R.S. (1995) 'Comparing predictive accuracy', *Journal of Business & Economic Statistics*, 13(3), pp. 253–263.
- Harvey, D., Leybourne, S. and Newbold, P. (1997) 'Testing the equality of prediction mean squared errors', *International Journal of Forecasting*, 13(2), pp. 281–291.
- Meese, R.A. and Rogoff, K. (1983) 'Empirical exchange rate models of the seventies: Do they fit out of sample?', *Journal of International Economics*, 14(1–2), pp. 3–24.
""")

footer()
