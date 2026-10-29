"""Data acquisition and cleaning for USD/VND.

Source priority (``mode="auto"``):
    1. Live daily quotes from Yahoo Finance (symbol ``VND=X``).
    2. Bundled CSV snapshot in ``data/`` (same source, frozen at a known date).
    3. Simulated series (only if both above fail, or when explicitly requested).

The returned object always states which source was used, so the UI can
label simulated data clearly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    HISTORY_START,
    SNAPSHOT_PATH,
    SPIKE_THRESHOLD,
    SPIKE_WINDOW,
    TICKER,
)


@dataclass(frozen=True)
class FXData:
    series: pd.Series      # cleaned close on a business-day index
    raw: pd.Series         # quotes as received (before cleaning)
    source: str            # "live" | "snapshot" | "simulated"
    label: str             # human-readable description for the UI
    n_cleaned: int         # number of quotes replaced by the spike filter

    @property
    def is_simulated(self) -> bool:
        return self.source == "simulated"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _standardise(s: pd.Series) -> pd.Series:
    """Float values, tz-naive sorted DatetimeIndex, no duplicates or non-positive quotes."""
    s = pd.to_numeric(s, errors="coerce")
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[s > 0].dropna()
    s.name = "close"
    s.index.name = "date"
    return s


def clean_spikes(
    s: pd.Series, window: int = SPIKE_WINDOW, threshold: float = SPIKE_THRESHOLD
) -> tuple[pd.Series, pd.Series]:
    """Replace isolated bad ticks with the previous valid quote.

    Free FX feeds for VND contain one-day spikes that fully revert the next day
    (e.g. +1.5% then -1.5%). They are not tradable prices and would dominate
    any volatility estimate. A quote is flagged when it deviates by more than
    ``threshold`` (log) from the centred rolling median. Genuine level shifts
    (e.g. a band widening) persist, so they move the median and are kept.

    Returns (cleaned_series, boolean_mask_of_flagged_points).
    """
    median = s.rolling(window, center=True, min_periods=3).median()
    flagged = (np.log(s / median)).abs() > threshold
    cleaned = s.mask(flagged).ffill().bfill()
    return cleaned, flagged


def to_business_days(s: pd.Series) -> pd.Series:
    """Reindex to a regular Mon–Fri calendar (forward-fill gaps such as holidays)."""
    idx = pd.bdate_range(s.index.min(), s.index.max(), name="date")
    out = s.reindex(idx).ffill()
    out.name = "close"
    return out


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def fetch_yahoo(ticker: str = TICKER, start: str = HISTORY_START, timeout: int = 15) -> pd.Series:
    """Download daily closes from Yahoo Finance. Raises on empty/failed download."""
    import yfinance as yf  # imported lazily so offline use does not need network

    df = yf.download(
        ticker, start=start, progress=False, auto_adjust=False, threads=False, timeout=timeout
    )
    if df is None or df.empty or "Close" not in df:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance >=0.2.4x returns MultiIndex columns
        close = close.iloc[:, 0]
    close = _standardise(close)
    if len(close) < 500:
        raise RuntimeError(f"Too few observations from Yahoo Finance ({len(close)})")
    return close


def load_snapshot(path=SNAPSHOT_PATH) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return _standardise(df["close"])


def save_snapshot(s: pd.Series, path=SNAPSHOT_PATH) -> None:
    out = s.rename("close").to_frame()
    out.index.name = "date"
    out.to_csv(path, float_format="%.2f")


def simulate_usdvnd(
    start: str = "2019-01-01", end: str | None = None, s0: float = 23_200.0, seed: int = 42
) -> pd.Series:
    """Synthetic USD/VND path for demos/tests. NOT market data.

    Stylised managed float: a slowly depreciating reference rate (~2.5%/yr),
    mean reversion of the spot towards it, GARCH(1,1) volatility clustering and
    rare step depreciations. Parameters are chosen to give daily volatility of
    roughly 0.1–0.3%, in the range of the cleaned real series.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end or pd.Timestamp.today().normalize(), name="date")
    n = len(idx)

    drift = np.log(1.025) / 252
    omega, alpha, beta = 2e-8, 0.08, 0.90  # GARCH(1,1); unconditional var = omega/(1-a-b)
    kappa = 0.02                            # daily pull towards the reference rate

    log_ref = np.log(s0) + drift * np.arange(n)
    log_s = np.empty(n)
    log_s[0] = np.log(s0)
    var = omega / (1 - alpha - beta)
    eps_prev = 0.0
    for t in range(1, n):
        var = omega + alpha * eps_prev**2 + beta * var
        eps = np.sqrt(var) * rng.standard_normal()
        jump = 0.008 if rng.random() < 0.003 else 0.0
        log_ref[t:] += jump  # step devaluations also move the reference rate
        log_s[t] = log_s[t - 1] + drift - kappa * (log_s[t - 1] - log_ref[t - 1]) + eps + jump
        eps_prev = eps

    return pd.Series(np.round(np.exp(log_s), 0), index=idx, name="close")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _bundle(raw: pd.Series, source: str, label: str) -> FXData:
    cleaned, flagged = clean_spikes(raw)
    return FXData(
        series=to_business_days(cleaned),
        raw=raw,
        source=source,
        label=label,
        n_cleaned=int(flagged.sum()),
    )


def load_fx_data(mode: str = "auto") -> FXData:
    """Load USD/VND according to ``mode``: "auto", "live", "snapshot" or "simulated"."""
    if mode == "simulated":
        sim = simulate_usdvnd()
        return FXData(sim, sim, "simulated", "Simulated data (synthetic, not market prices)", 0)

    errors = []
    if mode in ("auto", "live"):
        try:
            raw = fetch_yahoo()
            return _bundle(raw, "live", f"Yahoo Finance {TICKER}, live (to {raw.index.max():%d %b %Y})")
        except Exception as exc:  # network blocked, rate-limited, API change …
            errors.append(f"live: {exc}")
            if mode == "live":
                raise

    if mode in ("auto", "snapshot"):
        try:
            raw = load_snapshot()
            return _bundle(raw, "snapshot", f"Yahoo Finance {TICKER}, bundled snapshot (to {raw.index.max():%d %b %Y})")
        except Exception as exc:
            errors.append(f"snapshot: {exc}")
            if mode == "snapshot":
                raise

    sim = simulate_usdvnd()
    return FXData(sim, sim, "simulated",
                  "Simulated data (live feed and snapshot unavailable: " + "; ".join(errors) + ")", 0)
