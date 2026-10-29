# Treasury Analytics Lab – USD/VND and VND rates

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An interactive app covering the four building blocks of a bank Treasury quant role, on public
Vietnamese and US market data:

| Area | What the app does | Page |
|---|---|---|
| **Forecasting FX** | USD/VND 1–30 business days ahead: random walk, ARIMA, gradient boosting; walk-forward backtest with Diebold–Mariano tests | USD/VND forecast |
| **Forecasting rates** | VND government yield curve 1–6 months ahead with dynamic Nelson–Siegel (Diebold–Li); expanding-window backtest against the random walk | Rates & liquidity |
| **Curves & derivatives pricing** | Nelson–Siegel(–Svensson) zero curves calibrated to par yields; bond price, yield, duration, DV01, key-rate DV01, Z-spread; USD/VND forwards by covered interest parity; Garman–Kohlhagen FX options with Greeks | Curves & pricing |
| **Portfolio optimisation** | VND government bond mix maximising carry + roll-down for a given curve risk, under weight and duration limits; efficient frontier; stress tests incl. the worst historical 12-month move | Bond portfolio |
| **Data / SQL** | All sources cleaned once and loaded into an in-memory DuckDB warehouse; cross-dataset analysis (spreads, month-end alignment, rolling volatility, UIP check) written in SQL in `sql/` | SQL warehouse |

**Live demo:** https://YOUR-APP-NAME.streamlit.app  ← *replace after deploying (section 9)*

<!-- After deploying, add a screenshot as docs/screenshot.png and uncomment:
![App screenshot](docs/screenshot.png)
-->

---

## 1. Why these pieces, for a Treasury desk

| Desk question | Where it is answered |
|---|---|
| How far could USD/VND move against our position in *h* days, and is any model better than "no change"? | FX forecast + backtest |
| Where will VND government yields be in 1–6 months – and can a model beat the random walk? | Rates backtest |
| What is this bond worth on today's curve, how rich or cheap is a quoted price, and where is its rate risk? | Bond pricer (Z-spread, key-rate DV01) |
| What forward should we quote a client, and what does a USD call cost? | CIP forwards, Garman–Kohlhagen |
| Which tenors pay for their risk, and how bad is a 2022-style sell-off? | Bond portfolio frontier and stress tests |

The common thread is **honest evaluation**: every forecast is compared out of sample with a naive
benchmark, every calibration reports its fit error, and every conclusion states its limits.

## 2. Data

| Dataset | Source | Coverage in snapshot | Cleaning |
|---|---|---|---|
| USD/VND daily close | Yahoo Finance `VND=X` | Jan 2015 – Oct 2026 | One-day spikes (> 0.6% from the 11-day median) replaced; 157 of 3,076 quotes |
| VND government bond yields, 1Y–30Y, monthly | Vietnam Bond Market Association (VBMA), public secondary-market page | Jan 2020 – Aug 2026 (Apr–Dec 2020 not published) | Sep 2026 file identical to Aug 2026 → dropped; May 2023 short end inverts against both neighbouring months (1Y 4.90% between 2.70% and 2.30%) → month replaced by neighbours' average |
| US Treasury constant-maturity yields 1M–30Y and SOFR, daily | FRED (St. Louis Fed) | Jan 2014 – Oct 2026 | none |

Snapshots are stored as received in `data/`; cleaning happens in code and every change is listed in the app.
`VND=X` is an indicative quote, not the SBV central rate; USD/VND trades within ±5% of the central
rate since 17 October 2022 ([Tuoi Tre News](https://news.tuoitre.vn/vietnams-central-bank-widens-usd-vnd-exchange-rate-band-to-5-percent-10369617.htm)).

## 3. Methods

**FX forecasting.** Random walk (σ from 250 days), ARIMA(p,1,q) chosen by AIC, gradient boosting on
lagged returns with a recursive multi-step forecast and bootstrapped bands. Walk-forward backtest,
40 origins, 3-year window; Diebold–Mariano test with the Harvey–Leybourne–Newbold correction.

**Rates forecasting.** Dynamic Nelson–Siegel (Diebold and Li, 2006): level, slope and curvature
estimated each month by OLS with λ fixed (chosen on the training window only); direct h-step AR(1)
for each factor. Benchmarks: random walk and AR(1) tenor by tenor. Expanding window, 44 origins
(Dec 2022 – Jul 2026).

**Curve calibration.** Zero curve in Nelson–Siegel (VND, 8 tenors) or Svensson (USD, 11 tenors) form,
continuously compounded; parameters minimise the error between published **par** yields and the par
yields the curve implies (annual coupons for VND, semi-annual for USD). In the Svensson form
λ₂ = k·λ₁ with k ≤ 0.6; without it λ₁ ≈ λ₂ makes the two curvature terms collinear.

**Bonds.** ACT/ACT accrued interest, street-convention yield, modified duration, convexity, DV01;
key-rate DV01 from tent bumps of the zero curve (Ho, 1992), which add up to the parallel DV01;
Z-spread over the curve.

**FX forwards and options.** Covered interest parity from the two zero curves, and a money-market
version with simple interest; Garman–Kohlhagen (1983) price and Greeks; RiskMetrics EWMA volatility.

**Portfolio.** Par bond per tenor; expected return = carry + roll-down on an unchanged curve;
risk = key-rate DV01 × covariance of monthly yield changes; SLSQP with long-only, weight-cap and
duration constraints; efficient frontier; full-revaluation stress tests.

**SQL.** DuckDB views and queries in `sql/` (UNPIVOT, `arg_max` month-end alignment, window
functions, `FILTER` aggregates, `corr`/`regr_slope`). The in-app console is read-only: one statement,
file and network access disabled, configuration locked.

## 4. Key results

Snapshot to 23–27 Oct 2026 (VND curve to Aug 2026).

**FX.** Neither ARIMA nor gradient boosting beats the random walk out of sample: ARIMA is within 2%
of the benchmark at every horizon (not significant); gradient boosting is 41% worse at 30 days
(p = 0.03). The random-walk band is closest to its 90% coverage target.

**Rates.** The random walk is not beaten either. RMSE relative to the random walk (DM p-value):

| Horizon | Tenor | Random walk RMSE | DNS-AR(1) | AR(1) on yields |
|---:|---:|---:|---:|---:|
| 1 month | 2Y | 26 bp | 1.06× (0.43) | 1.18× (0.06) |
| 1 month | 10Y | 19 bp | 1.53× (0.00) | 1.15× (0.08) |
| 3 months | 10Y | 47 bp | 1.74× (0.17) | 1.50× (0.21) |
| 6 months | 10Y | 72 bp | 1.41× (0.05) | 2.13× (0.27) |

DNS-AR(1) is significantly worse than the random walk in 7 of 24 tenor–horizon cells and never
significantly better. Both models pull yields towards their sample mean, which costs accuracy in a
sample with clear trends (yields fell through 2023 and rose through 2025–26). With 44 overlapping
monthly origins the tests have little power.

**Curves.** Nelson–Siegel fits the VBMA curves with a median RMSE of 6.6 bp (90th percentile 11.7 bp,
68 months); Svensson fits the US curve of 27 Oct 2026 within 3.2 bp.

**Forwards.** On government curves the VND 1-year zero rate (3.47%) is below the US one (4.47%), so
parity puts the 1-year USD/VND forward about 258 VND *below* spot (25,848). Interbank VND rates are
higher than government yields, so dealer forward points, which come from money-market and FX-swap
curves, will differ; the app shows the government-curve line as a lower reference, not a quote.

**Portfolio (12-month horizon, 40% cap).** Carry + roll-down rises from 3.53% (1Y) to 4.78% (10Y) and
then stays flat to 30Y while risk keeps rising: the frontier goes flat beyond roughly 5% risk / duration 6.
An equal-weight portfolio loses 6.4% under a +100 bp parallel shift and 18.0% under the Nov 2021 →
Nov 2022 move (10Y +281 bp).

**UIP.** The VN–US 1-year yield gap does not predict the next 12 months of USD/VND in this sample:
regression slope −0.72 (parity implies +1), correlation −0.38, 59 overlapping windows.

## 5. Tech stack

Python · pandas · NumPy · SciPy (least squares, SLSQP, root finding) · statsmodels · scikit-learn ·
DuckDB · Plotly · Streamlit · pytest

## 6. Repository structure

```
fx-forecast-app/
├── app.py                      # entry point: page navigation
├── views/                      # one file per page
│   ├── fx_forecast.py          #   USD/VND forecast and backtest
│   ├── rates_liquidity.py      #   VND curve forecast and backtest; interbank liquidity
│   ├── pricing.py              #   curve calibration, bond pricer, forwards, FX options
│   ├── portfolio.py            #   bond portfolio optimisation and stress tests
│   └── sql_warehouse.py        #   DuckDB schema, named queries, read-only console
├── src/
│   ├── config.py, data_loader.py, models.py, backtest.py, plotting.py   # FX
│   ├── rates_data.py           # VBMA / FRED download and cleaning
│   ├── curves.py               # Nelson–Siegel(–Svensson) calibration, key-rate bumps, scenarios
│   ├── rates_forecast.py       # dynamic Nelson–Siegel, curve backtest
│   ├── pricing.py              # bonds, CIP forwards, Garman–Kohlhagen, EWMA vol
│   ├── portfolio.py            # carry/roll-down universe, optimiser, frontier, stress tests
│   ├── warehouse.py            # DuckDB build and query sandbox
│   ├── rates_plotting.py, app_common.py
├── sql/                        # views and named queries run by the warehouse
├── scripts/refresh_data.py     # refresh FX snapshot + precomputed FX backtest
├── scripts/refresh_rates.py    # refresh VBMA and FRED snapshots
├── data/                       # snapshots (see data/README.md)
├── notebooks/01_eda_and_backtest.ipynb
├── tests/                      # pytest: test_core.py (FX), test_rates.py (rates, pricing, portfolio, SQL)
├── requirements.txt, requirements-dev.txt
└── README.md
```

## 7. Run locally

```bash
git clone https://github.com/nducngocneu-cyber/fx-forecast-app.git
cd fx-forecast-app
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

Optional:

```bash
pip install -r requirements-dev.txt
pytest -q                          # 32 unit tests
python -m scripts.refresh_data     # latest USD/VND + FX backtest (~20 s)
python -m scripts.refresh_rates    # latest VBMA and FRED yields
```

## 8. Limitations

- Short samples: monthly VND yields from 2021 (68 months); results describe this period.
- VBMA reference yields are coupon-bond levels at generic tenors, not traded bond lines, and have no
  tenor below one year, so the VND government curve is extrapolated below 1Y.
- Portfolio analysis uses generic par bonds, ignores bid-ask, liquidity and repo funding, and its
  expected return is the unchanged-curve scenario, not a forecast.
- Log-normal option prices ignore the SBV trading band and devaluation steps.
- Forward points from government curves are not checked against traded forward quotes.

## 9. Deploy on Streamlit Community Cloud (free)

1. Edit `AUTHOR` and `REPO_URL` in `src/app_common.py`, then push the repository to GitHub (public).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Select the repository, branch `main`, main file `app.py`. Under *Advanced settings* choose Python 3.12.
4. Optionally set a custom subdomain, then **Deploy**.
5. Copy the app URL into the badge and *Live demo* line above, and into your CV.

The free tier sleeps after a period without traffic and wakes on the next visit. Yahoo Finance sometimes
rate-limits shared cloud IPs; the FX page then falls back to the bundled snapshot and says so.

## References

Diebold, F.X. and Li, C. (2006) 'Forecasting the term structure of government bond yields', *Journal of Econometrics*, 130(2), pp. 337–364.

Diebold, F.X. and Mariano, R.S. (1995) 'Comparing predictive accuracy', *Journal of Business & Economic Statistics*, 13(3), pp. 253–263.

Garman, M.B. and Kohlhagen, S.W. (1983) 'Foreign currency option values', *Journal of International Money and Finance*, 2(3), pp. 231–237.

Harvey, D., Leybourne, S. and Newbold, P. (1997) 'Testing the equality of prediction mean squared errors', *International Journal of Forecasting*, 13(2), pp. 281–291.

Ho, T.S.Y. (1992) 'Key rate durations: Measures of interest rate risks', *Journal of Fixed Income*, 2(2), pp. 29–44.

Meese, R.A. and Rogoff, K. (1983) 'Empirical exchange rate models of the seventies: Do they fit out of sample?', *Journal of International Economics*, 14(1–2), pp. 3–24.

Nelson, C.R. and Siegel, A.F. (1987) 'Parsimonious modeling of yield curves', *Journal of Business*, 60(4), pp. 473–489.

Svensson, L.E.O. (1994) 'Estimating and interpreting forward interest rates: Sweden 1992–1994', NBER Working Paper No. 4871.

---

**Disclaimer:** educational portfolio project; not investment advice.
**Author:** Nguyen Duc Ngoc · [LinkedIn](https://www.linkedin.com/in/<your-profile>)
