"""Refresh the interest-rate snapshots.

Usage (from the repo root):
    python -m scripts.refresh_rates

Outputs:
    data/vn_govt_yields_monthly.csv   VBMA monthly reference yields, Vietnam, as published (long format)
    data/us_treasury_daily.csv        FRED daily Treasury CMT yields 1M–30Y and SOFR (percent)
"""
from __future__ import annotations

from src.rates_data import US_YIELDS_PATH, VN_YIELDS_PATH, fetch_fred, fetch_vbma, load_vn_yields


def main() -> None:
    vn = fetch_vbma(start="2018-01")
    vn.to_csv(VN_YIELDS_PATH, index=False)
    h = load_vn_yields()
    print(f"VBMA: {vn['month'].nunique()} monthly files, {h.yields.index.min():%Y-%m} → "
          f"{h.yields.index.max():%Y-%m}; {len(h.flags)} cleaning flags")

    us = fetch_fred(start="2014-01-01")
    us.to_csv(US_YIELDS_PATH, float_format="%.2f")
    print(f"FRED: {len(us)} days to {us.dropna(how='all').index.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
