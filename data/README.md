# Data

| File | Content |
|---|---|
| `usdvnd_daily.csv` | Daily closing quotes of USD/VND (VND per 1 USD), Yahoo Finance symbol `VND=X`, from 2015-01-01. Stored **raw** (before cleaning) so the cleaning step is reproducible. |
| `backtest_results.csv` | Walk-forward backtest output: one row per model × origin × horizon (spot, forecast, interval, actual, error). |
| `backtest_meta.json` | Settings and date range of the precomputed backtest. |

Refresh both with:

```bash
python -m scripts.refresh_data
```

**Notes on the source.** `VND=X` is an aggregated, indicative quote. It is not a State Bank of
Vietnam (SBV) central rate and not a tradable interbank price; it contains isolated bad ticks,
which `src/data_loader.clean_spikes` removes. The app uses live Yahoo data when reachable and
falls back to this snapshot. A **simulated** series (`simulate_usdvnd`) exists for offline demos
and tests; the app labels it clearly whenever it is used.

## Interest-rate snapshots

| File | Content |
|---|---|
| `vn_govt_yields_monthly.csv` | Vietnam government bond reference yields by tenor (1Y–30Y), one row per month × tenor, as published by the Vietnam Bond Market Association (VBMA) on its public secondary-market page (`vbma.org.vn/csv/markets/charts/en/thucap/MM_YYYY.csv`). Stored as published; `src/rates_data.clean_vn_yields` drops a stale month and repairs a reverting spike (May 2023). |
| `us_treasury_daily.csv` | US Treasury constant-maturity yields (FRED series DGS1MO … DGS30) and SOFR, daily, percent. |

Refresh both with `python -m scripts.refresh_rates`.
