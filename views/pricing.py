"""Page: curve calibration, bond pricing and risk, USD/VND forwards and FX options."""
import numpy as np
import pandas as pd
import streamlit as st

from src.app_common import us_curve, vn_curve, vn_history
from src.app_common import footer
from src.data_loader import load_fx_data
from src.pricing import (Bond, cip_forward, cip_forward_money_market, ewma_vol, garman_kohlhagen)
from src.rates_plotting import (AMBER, BLUE, bar_figure, curve_snapshot_figure, forward_points_figure,
                                option_value_figure)

st.title("Curves & pricing")
st.markdown("Zero curves calibrated to published government yields, then used to price and risk-manage "
            "bonds, USD/VND forwards and FX options.")

yields, _, _, _ = vn_history()
vn_asof = yields.index[-1]
cv = vn_curve(vn_asof)
us_date, us_obs, cu = us_curve()


@st.cache_data(show_spinner=False)
def fx_snapshot():
    d = load_fx_data("snapshot")
    return d.series, d.label


fx, fx_label = fx_snapshot()
spot, spot_date = float(fx.iloc[-1]), fx.index[-1]
tab_curve, tab_bond, tab_fwd, tab_opt, tab_method = st.tabs(
    ["📈 Curve calibration", "🏷️ Bond pricer", "🔁 USD/VND forwards", "🎛️ FX options", "📘 Methodology"])

# --------------------------------------------------------------------------- calibration
with tab_curve:
    vn_obs = yields.loc[vn_asof]
    st.plotly_chart(curve_snapshot_figure([
        (f"VND {vn_asof:%b %Y}", BLUE, vn_obs.index.to_numpy(float), vn_obs.to_numpy(), cv),
        (f"USD {us_date:%d %b %Y}", AMBER, us_obs.index.to_numpy(float), us_obs.to_numpy(), cu),
    ]), width="stretch")
    rows = []
    for name, c in [("VND government (Nelson–Siegel)", cv), ("US Treasury (Svensson)", cu)]:
        p = c.params
        rows.append({"Curve": name, "β0 level (%)": p["beta0"] * 100, "β1 slope (%)": p["beta1"] * 100,
                     "β2 curvature (%)": p["beta2"] * 100, "β3 (%)": p["beta3"] * 100,
                     "λ1": p["lam1"], "λ2": p["lam2"] if c.model == "NSS" else np.nan,
                     "Fit RMSE (bp)": c.rmse_bp, "Max |error| (bp)": np.max(np.abs(c.fit_model - c.fit_observed)) * 100})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    grid = np.array([1, 2, 3, 5, 7, 10, 20, 30], float)
    zt = pd.DataFrame({"Maturity": [f"{t:g}Y" for t in grid],
                       "VND zero (%)": cv.zero(grid) * 100, "USD zero (%)": cu.zero(grid) * 100,
                       "VND 1Y forward starting here (%)": cv.forward(grid, grid + 1) * 100,
                       "VND discount factor": cv.df(grid)})
    st.dataframe(zt.round(4), hide_index=True, width="stretch")
    st.caption("Zero rates are continuously compounded. The VND curve has 8 published tenors, so the "
               "4-parameter Nelson–Siegel form is used; Svensson's two extra parameters ended at their bounds "
               "in most months when tried, a sign of over-fitting.")

# --------------------------------------------------------------------------- bond pricer
with tab_bond:
    st.markdown(f"Fixed-coupon VND government bond, annual coupon, valued on the VND curve of **{vn_asof:%d %b %Y}**.")
    c1, c2, c3, c4 = st.columns(4)
    settle = pd.Timestamp(c1.date_input("Settlement", value=vn_asof.date()))
    maturity = pd.Timestamp(c2.date_input("Maturity", value=(vn_asof + pd.DateOffset(years=10)).date()))
    coupon = c3.number_input("Coupon (% a year)", 0.0, 20.0, 4.50, 0.05)
    quote = c4.number_input("Market clean price (blank = price off curve)", 0.0, 200.0, 0.0, 0.01,
                            help="Enter a traded price to get its yield and Z-spread; 0 uses the curve price.")
    if maturity <= settle:
        st.error("Maturity must be after settlement.")
    else:
        b = Bond(maturity, coupon, freq=1)
        curve_dirty = b.dirty_from_curve(settle, cv)
        acc = b.accrued(settle)
        clean = quote if quote > 0 else curve_dirty - acc
        y = b.ytm(settle, clean)
        risk = b.risk_from_yield(settle, y)
        zs = b.z_spread(settle, cv, clean)
        m = st.columns(6)
        m[0].metric("Clean price", f"{clean:.3f}")
        m[1].metric("Accrued", f"{acc:.3f}")
        m[2].metric("Yield to maturity", f"{y:.3f}%")
        m[3].metric("Modified duration", f"{risk['modified']:.2f}")
        m[4].metric("DV01 per 100 face", f"{risk['dv01']:.4f}")
        m[5].metric("Z-spread vs curve", f"{zs:+.1f} bp")
        st.caption(f"Convexity {risk['convexity']:.2f}; Macaulay duration {risk['macaulay']:.2f} years; "
                   f"curve price (clean) {curve_dirty - acc:.3f}. DV01 for 10 bn VND face value: "
                   f"{risk['dv01'] / 100 * 10e9:,.0f} VND per bp.")
        tenors = list(yields.columns)
        krd = b.key_rate_dv01(settle, cv, tenors)
        st.markdown("**Key-rate DV01** – price fall per 100 face for a 1 bp move at each tenor "
                    "(tent bumps, Ho 1992). The bars add up to the parallel DV01 on the curve.")
        st.plotly_chart(bar_figure([f"{t:g}Y" for t in krd.index], krd.to_numpy(), "DV01 per 100 face",
                                   BLUE, ".4f"), width="stretch")
        with st.expander("Cash-flow schedule"):
            cf = b.cashflows(settle)
            cf["discount factor"] = cv.df(cf["t"].to_numpy())
            cf["PV"] = cf["cf"] * cf["discount factor"]
            st.dataframe(cf.round({"t": 4, "cf": 4, "discount factor": 6, "PV": 4}), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- forwards
with tab_fwd:
    st.markdown(
        f"Covered interest parity: the outright forward that leaves no arbitrage between borrowing one "
        f"currency and lending the other. Spot **{spot:,.0f}** ({spot_date:%d %b %Y}, {fx_label})."
    )
    tenor_days = {"1M": 30, "3M": 91, "6M": 182, "9M": 273, "1Y": 365}
    t_years = np.array(list(tenor_days.values())) / 365
    gov = cip_forward(spot, t_years, cv, cu) - spot
    df = pd.DataFrame({"tenor_label": list(tenor_days), "Government curves (VND gov vs UST)": gov})
    ib = None
    try:
        from src.interbank import latest_interbank_curve
        ib_date, ib_rates = latest_interbank_curve()
        # USD money-market rate (simple, ACT/360) with the same discount factor as the Treasury curve
        usd_mm = {k: float((1 / cu.df(np.array([d / 365]))[0] - 1) * 360 / d * 100) for k, d in tenor_days.items()}
        vnd_mm = {k: ib_rates.get(k) for k in tenor_days}
        mm = [cip_forward_money_market(spot, d, vnd_mm[k], usd_mm[k]) - spot if vnd_mm[k] is not None else np.nan
              for k, d in tenor_days.items()]
        df["Interbank VND vs US bills"] = mm
        ib = ib_date
    except (ImportError, FileNotFoundError):
        pass
    st.plotly_chart(forward_points_figure(df), width="stretch")
    show = df.rename(columns={"tenor_label": "Tenor"}).copy()
    for col in show.columns[1:]:
        show[col.replace("points", "") + " outright"] = spot + show[col]
    st.dataframe(show.round(1), hide_index=True, width="stretch")
    st.markdown(
        "**Reading the gap.** With government curves, VND yields sit *below* US Treasury yields at the short "
        "end, so parity puts USD/VND forwards *below* spot. Interbank VND rates are higher than "
        "government yields (they carry bank credit and liquidity premia), which pushes forward points up. "
        "Dealers price FX forwards off money-market and FX-swap curves, so the interbank line is the "
        "closer proxy (each interbank tenor is its volume-weighted average over the last five reporting days, "
        "because the 6M and 9M tenors trade thinly); neither line has been checked against traded forward quotes. VBMA publishes no "
        "VND government tenor below one year, so the government line below 1Y uses the extrapolated curve."
        + ("" if ib is not None else " *Interbank data is not loaded in this build.*")
    )

# --------------------------------------------------------------------------- options
with tab_opt:
    log_ret = np.log(fx).diff().dropna()
    ewma = ewma_vol(log_ret)
    c1, c2, c3, c4, c5 = st.columns(5)
    kind = c1.radio("Type", ["call", "put"], horizontal=True,
                    help="Call = right to buy USD (sell VND) at the strike.")
    days = c2.number_input("Days to expiry", 7, 730, 91)
    T = days / 365
    fwd = float(cip_forward(spot, [T], cv, cu)[0])
    strike = c3.number_input("Strike (VND per USD)", 10_000.0, 60_000.0, float(round(spot * 1.01, -1)), 10.0)
    vol = c4.number_input("Volatility (% a year)", 0.1, 50.0, float(round(ewma, 2)), 0.1,
                          help="Default: RiskMetrics EWMA (λ = 0.94) of daily USD/VND log returns.")
    notional = c5.number_input("Notional (USD)", 10_000, 100_000_000, 1_000_000, 10_000)
    r_vnd = float(cv.zero([T])[0] * 100)
    r_usd = float(cu.zero([T])[0] * 100)
    g = garman_kohlhagen(spot, strike, T, r_vnd, r_usd, vol, kind)
    m = st.columns(6)
    m[0].metric("Premium (VND per USD)", f"{g['price']:,.1f}")
    m[1].metric("Premium (% of spot)", f"{g['price'] / spot * 100:.3f}%")
    m[2].metric("Delta", f"{g['delta']:.3f}")
    m[3].metric("Vega per 1 vol pt", f"{g['vega']:,.1f}")
    m[4].metric("Theta per day", f"{g['theta']:,.2f}")
    m[5].metric("Premium on notional", f"{g['price'] * notional / 1e6:,.1f} m VND")
    st.caption(f"Rates (continuous, from the calibrated curves at {days} days): VND {r_vnd:.2f}%, USD {r_usd:.2f}%; "
               f"parity forward {g['forward']:,.0f}. Gamma {g['gamma']:.2e} per VND; rho VND {g['rho_dom']:,.1f}, "
               f"rho USD {g['rho_for']:,.1f} per 1 pp. EWMA vol today {ewma:.2f}%.")
    spots = np.linspace(spot * 0.94, spot * 1.06, 121)
    vals = [garman_kohlhagen(s, strike, T, r_vnd, r_usd, vol, kind)["price"] for s in spots]
    payoff = np.maximum(spots - strike, 0) if kind == "call" else np.maximum(strike - spots, 0)
    st.plotly_chart(option_value_figure(spots, vals, payoff, spot, kind), width="stretch")
    st.info("USD/VND is a managed currency: it trades within a band around the State Bank of Vietnam's central "
            "rate (±5% since 17 October 2022) and moves in steps around policy changes. Log-normal "
            "Garman–Kohlhagen prices ignore both, so they understate the cost of protection against a "
            "devaluation step and overstate the value of strikes beyond the band edge.")

# --------------------------------------------------------------------------- method
with tab_method:
    st.markdown(r"""
#### Curve calibration
Published yields are par yields of coupon bonds. The zero curve $z(t)$ (continuous compounding) has a
Nelson–Siegel or Svensson form, and its parameters minimise the squared error between published par yields
and the par yields the curve implies:
$$\text{par}(T) = f\,\frac{1 - D(T)}{\sum_i D(t_i)}, \qquad D(t) = e^{-z(t)\,t}$$
with $f$ = 1 coupon a year for Vietnamese government bonds and 2 for US Treasuries. Several starting values for
the decay parameters are tried because the problem is not convex in them.

#### Bonds
Coupon schedule rolled back from maturity; accrued interest ACT/ACT; yield with annual compounding (street
convention). DV01 = modified duration × dirty price × 1 bp. Key-rate DV01 use triangular bumps of the zero
curve at the published tenors (Ho, 1992), which sum to a parallel shift. The Z-spread is the constant spread
over the zero curve that reproduces a quoted price.

#### FX forwards and options
Covered interest parity: $F = S \cdot D_{USD}(T) / D_{VND}(T)$, or with money-market simple interest
$F = S \cdot (1 + r_{VND} d/365)/(1 + r_{USD} d/360)$. Options: Garman and Kohlhagen (1983), the
Black–Scholes model with a foreign interest rate.

#### References
- Garman, M.B. and Kohlhagen, S.W. (1983) 'Foreign currency option values', *Journal of International Money and Finance*, 2(3), pp. 231–237.
- Ho, T.S.Y. (1992) 'Key rate durations: Measures of interest rate risks', *Journal of Fixed Income*, 2(2), pp. 29–44.
- Nelson, C.R. and Siegel, A.F. (1987) 'Parsimonious modeling of yield curves', *Journal of Business*, 60(4), pp. 473–489.
- Svensson, L.E.O. (1994) 'Estimating and interpreting forward interest rates: Sweden 1992–1994', NBER Working Paper 4871.
""")

footer()
