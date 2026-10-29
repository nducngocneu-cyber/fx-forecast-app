"""Interest-rate data: Vietnamese and US government yield curves.

Sources
-------
* **Vietnam government bond yields** – monthly reference yields by tenor (1Y–30Y)
  published by the Vietnam Bond Market Association (VBMA) on its public
  "secondary market" page, one tab-separated UTF-16 file per month.
* **US Treasury yields** – daily constant-maturity yields (1M–30Y) and SOFR from
  FRED (Federal Reserve Bank of St. Louis), series DGS1MO … DGS30 and SOFR.

Everything downloaded is stored *as received* in ``data/`` so that the cleaning
steps below are reproducible offline.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DATA_DIR

VN_YIELDS_PATH = DATA_DIR / "vn_govt_yields_monthly.csv"
US_YIELDS_PATH = DATA_DIR / "us_treasury_daily.csv"

VBMA_URL = "https://vbma.org.vn/csv/markets/charts/en/thucap/{mm}_{yyyy}.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"

# Tenor label -> maturity in years
VN_TENORS = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30}
US_SERIES = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1, "DGS2": 2, "DGS3": 3,
    "DGS5": 5, "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30,
}

# A monthly yield is treated as a bad print when it differs from BOTH neighbours
# by more than this and the neighbours agree with each other (it reverts).
SPIKE_BP = 100


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def parse_vbma_file(raw: bytes, month: str) -> pd.DataFrame:
    """Parse one VBMA monthly file into long rows (month, country, tenor, yield_pct)."""
    for enc in ("utf-16", "utf-8-sig"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode VBMA file for {month}")
    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    header = [h.strip() for h in lines[0].split("\t")]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split("\t")]
        for tenor, val in zip(header[1:], parts[1:]):
            try:
                rows.append({"month": month, "country": parts[0], "tenor": tenor,
                             "yield_pct": float(val)})
            except ValueError:
                continue  # blank cells occur in a few files
    return pd.DataFrame(rows)


def fetch_vbma(start: str = "2018-01", end: str | None = None, timeout: int = 15) -> pd.DataFrame:
    """Download every available monthly VBMA file between ``start`` and ``end``."""
    import requests

    end = end or pd.Timestamp.today().strftime("%Y-%m")
    frames = []
    for p in pd.period_range(start, end, freq="M"):
        url = VBMA_URL.format(mm=f"{p.month:02d}", yyyy=p.year)
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            continue  # months VBMA did not publish
        frames.append(parse_vbma_file(r.content, str(p)))
    if not frames:
        raise RuntimeError("No VBMA files downloaded")
    df = pd.concat(frames, ignore_index=True)
    return df[df["country"] == "VN"].drop(columns="country").reset_index(drop=True)


def fetch_fred(start: str = "2014-01-01", timeout: int = 30) -> pd.DataFrame:
    """Daily US Treasury CMT yields and SOFR from FRED, wide format (percent)."""
    import requests

    cols = {}
    for series in [*US_SERIES, "SOFR"]:
        r = requests.get(FRED_URL.format(series=series, start=start), timeout=timeout)
        r.raise_for_status()
        s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True).iloc[:, 0]
        cols[series] = pd.to_numeric(s, errors="coerce")
    out = pd.DataFrame(cols)
    out.index.name = "date"
    return out


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CurveHistory:
    yields: pd.DataFrame        # index = month-end Timestamp, columns = maturity in years, %
    raw: pd.DataFrame           # same shape, as published
    flags: pd.DataFrame         # issues found: date, tenor, issue, published, used
    source: str


def clean_vn_yields(long: pd.DataFrame) -> CurveHistory:
    """Pivot VBMA rows to a month × tenor table and repair two known defects.

    1. **Stale months** – a month whose every tenor equals the previous month to
       the third decimal is a carried-forward file, not new data; it is dropped.
    2. **Reverting spikes** – a single tenor that jumps ≥ ``SPIKE_BP`` away from
       both neighbouring months while those neighbours are close to each other
       is replaced by the average of the neighbours (e.g. 1Y in May 2023).
    """
    wide = (long.assign(tenor_y=long["tenor"].map(VN_TENORS))
                .pivot_table(index="month", columns="tenor_y", values="yield_pct"))
    wide.index = pd.PeriodIndex(wide.index, freq="M").to_timestamp(how="end").normalize()
    wide = wide.sort_index()
    raw = wide.copy()
    flags = []

    same_as_prev = (wide.diff().abs() < 1e-9).all(axis=1)
    for d in wide.index[same_as_prev]:
        flags.append({"date": d, "tenor": "all", "issue": "identical to previous month (stale file)",
                      "published": np.nan, "used": np.nan})
    wide = wide[~same_as_prev]

    # Spike test only between consecutive calendar months (gaps break the comparison).
    vals = wide.copy()
    month_no = wide.index.year * 12 + wide.index.month
    consecutive = pd.Series(month_no, index=wide.index)
    has_both = ((consecutive - consecutive.shift(1)) == 1) & ((consecutive.shift(-1) - consecutive) == 1)
    prev, nxt = wide.shift(1), wide.shift(-1)
    bad = (((wide - prev).abs() * 100 >= SPIKE_BP) & ((wide - nxt).abs() * 100 >= SPIKE_BP)
           & ((prev - nxt).abs() * 100 < SPIKE_BP / 2))
    bad_months = bad.any(axis=1) & has_both
    # If any tenor spikes, the whole month's file is suspect (in May 2023 the short end
    # inverts against both neighbours), so every tenor is replaced by the neighbours' mean.
    for d in wide.index[bad_months]:
        spiking = [f"{c}Y" for c in wide.columns if bad.loc[d, c]]
        repl = (prev.loc[d] + nxt.loc[d]) / 2
        flags.append({"date": d, "tenor": ", ".join(spiking) + " (whole month replaced)",
                      "issue": "one-month spike that reverts",
                      "published": "; ".join(f"{c}Y {wide.loc[d, c]:.2f}" for c in wide.columns),
                      "used": "; ".join(f"{c}Y {repl[c]:.2f}" for c in wide.columns)})
        vals.loc[d] = repl.to_numpy()
    flags_df = pd.DataFrame(flags, columns=["date", "tenor", "issue", "published", "used"])
    return CurveHistory(vals, raw, flags_df, "VBMA monthly reference yields")


def continuous_tail(df: pd.DataFrame) -> pd.DataFrame:
    """Longest run of consecutive months ending at the last observation (for time-series models)."""
    m = df.index.year * 12 + df.index.month
    breaks = np.where(np.diff(m) != 1)[0]
    start = breaks[-1] + 1 if len(breaks) else 0
    return df.iloc[start:]


def load_vn_yields(path=VN_YIELDS_PATH) -> CurveHistory:
    return clean_vn_yields(pd.read_csv(path, dtype={"month": str}))


def load_us_yields(path=US_YIELDS_PATH) -> pd.DataFrame:
    """Daily US curve: columns are maturities in years (plus 'SOFR'), percent."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    curve = df[list(US_SERIES)].rename(columns=US_SERIES).dropna(how="all")
    curve["SOFR"] = df["SOFR"]
    return curve


def latest_curve(curve: pd.DataFrame, asof: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Series]:
    """Most recent complete row on or before ``asof`` (maturity columns only)."""
    mats = [c for c in curve.columns if not isinstance(c, str)]
    c = curve[mats].dropna()
    if asof is not None:
        c = c[c.index <= asof]
    return c.index[-1], c.iloc[-1]
