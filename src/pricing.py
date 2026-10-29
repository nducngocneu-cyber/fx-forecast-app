"""Pricing: fixed-coupon bonds, FX forwards (covered interest parity) and FX options.

Conventions (stated, because each one changes the numbers):
* Bonds: fixed coupon paid ``freq`` times a year, schedule rolled back from maturity,
  accrued interest ACT/ACT (ICMA), yield-to-maturity with ``freq`` compounding (street
  convention). Vietnamese government bonds pay annual coupons (freq=1).
* Curve discounting: continuously compounded zero rates, time in years ACT/365.
* FX: USD/VND quoted as VND per 1 USD. USD is the foreign (base) currency, VND the
  domestic (quote) currency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from src.curves import ZeroCurve, key_rate_bump


# --------------------------------------------------------------------------- #
# Bonds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bond:
    maturity: pd.Timestamp
    coupon_pct: float          # annual coupon rate, %
    freq: int = 1
    face: float = 100.0

    def schedule(self, settle: pd.Timestamp) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
        """Coupon dates after ``settle`` and the coupon date on or before it."""
        months = 12 // self.freq
        dates = [pd.Timestamp(self.maturity)]
        while dates[-1] > settle:
            dates.append(dates[-1] - pd.DateOffset(months=months))
        prev = dates[-1]
        future = pd.DatetimeIndex(sorted(d for d in dates if d > settle))
        return future, prev

    def cashflows(self, settle: pd.Timestamp) -> pd.DataFrame:
        settle = pd.Timestamp(settle)
        dates, _ = self.schedule(settle)
        cf = np.full(len(dates), self.coupon_pct / self.freq * self.face / 100)
        cf[-1] += self.face
        t = (dates - settle).days / 365.0
        return pd.DataFrame({"date": dates, "t": t, "cf": cf})

    def accrued(self, settle: pd.Timestamp) -> float:
        settle = pd.Timestamp(settle)
        dates, prev = self.schedule(settle)
        nxt = dates[0]
        frac = (settle - prev).days / (nxt - prev).days
        return self.coupon_pct / self.freq * self.face / 100 * frac

    # --- yield-based ------------------------------------------------------- #
    def _period_times(self, settle) -> tuple[np.ndarray, np.ndarray]:
        settle = pd.Timestamp(settle)
        dates, prev = self.schedule(settle)
        w = (dates[0] - settle).days / (dates[0] - prev).days       # fraction of first period left
        k = np.arange(len(dates)) + w
        cf = self.cashflows(settle)["cf"].to_numpy()
        return k, cf

    def dirty_from_yield(self, settle, ytm_pct: float) -> float:
        k, cf = self._period_times(settle)
        return float(np.sum(cf / (1 + ytm_pct / 100 / self.freq) ** k))

    def clean_from_yield(self, settle, ytm_pct: float) -> float:
        return self.dirty_from_yield(settle, ytm_pct) - self.accrued(settle)

    def ytm(self, settle, clean_price: float) -> float:
        dirty = clean_price + self.accrued(settle)
        return brentq(lambda y: self.dirty_from_yield(settle, y) - dirty, -5.0, 50.0, xtol=1e-10)

    def risk_from_yield(self, settle, ytm_pct: float) -> dict:
        """Macaulay and modified duration, convexity and DV01 (per 100 face) at a given yield."""
        k, cf = self._period_times(settle)
        y = ytm_pct / 100 / self.freq
        pv = cf / (1 + y) ** k
        price = pv.sum()
        mac = np.sum(k * pv) / price / self.freq
        mod = mac / (1 + y)
        conv = np.sum(cf * k * (k + 1) / (1 + y) ** (k + 2)) / price / self.freq**2
        return {"dirty": price, "macaulay": mac, "modified": mod, "convexity": conv,
                "dv01": mod * price * 1e-4}

    # --- curve-based ------------------------------------------------------- #
    def dirty_from_curve(self, settle, curve: ZeroCurve, spread_bp: float = 0.0) -> float:
        c = self.cashflows(settle)
        z = curve.zero(c["t"].to_numpy()) + spread_bp / 1e4
        return float(np.sum(c["cf"].to_numpy() * np.exp(-z * c["t"].to_numpy())))

    def z_spread(self, settle, curve: ZeroCurve, clean_price: float) -> float:
        """Constant spread (bp) over the zero curve that reproduces the market price."""
        dirty = clean_price + self.accrued(settle)
        return brentq(lambda s: self.dirty_from_curve(settle, curve, s) - dirty, -2000, 5000, xtol=1e-8)

    def key_rate_dv01(self, settle, curve: ZeroCurve, key_tenors) -> pd.Series:
        """Price fall (per 100 face) for a 1 bp tent bump at each key tenor."""
        base = self.dirty_from_curve(settle, curve)
        out = {}
        for k in key_tenors:
            bumped = curve.shifted(key_rate_bump(k, key_tenors, 1.0))
            out[k] = base - self.dirty_from_curve(settle, bumped)
        return pd.Series(out, name="KR DV01")


def par_bond(settle: pd.Timestamp, years: float, curve: ZeroCurve, freq: int = 1) -> Bond:
    """Bond maturing ``years`` after settle with the curve's par coupon (prices at ~100)."""
    coupon = float(curve.par_yield([years], freq)[0])
    mat = pd.Timestamp(settle) + pd.DateOffset(months=int(round(years * 12)))
    return Bond(mat, coupon, freq)


# --------------------------------------------------------------------------- #
# FX forwards
# --------------------------------------------------------------------------- #
def cip_forward(spot: float, t, vnd_curve: ZeroCurve, usd_curve: ZeroCurve) -> np.ndarray:
    """Outright USD/VND forward from two zero curves: F = S · DF_USD(t) / DF_VND(t)."""
    t = np.asarray(t, float)
    return spot * usd_curve.df(t) / vnd_curve.df(t)


def cip_forward_money_market(spot: float, days, r_vnd_pct, r_usd_pct) -> np.ndarray:
    """Money-market CIP with simple interest: VND ACT/365, USD ACT/360.

    F = S · (1 + r_VND·d/365) / (1 + r_USD·d/360)
    """
    d = np.asarray(days, float)
    return spot * (1 + np.asarray(r_vnd_pct) / 100 * d / 365) / (1 + np.asarray(r_usd_pct) / 100 * d / 360)


# --------------------------------------------------------------------------- #
# FX options – Garman & Kohlhagen (1983)
# --------------------------------------------------------------------------- #
def garman_kohlhagen(spot, strike, t, r_dom_pct, r_for_pct, vol_pct, kind: str = "call") -> dict:
    """Price and Greeks of a European option on 1 unit of foreign currency (USD).

    r_dom = VND rate, r_for = USD rate (continuously compounded, %), vol in % a year.
    Price, delta and gamma are in VND per USD notional; vega per 1 vol point; theta per
    calendar day; rho per 1 percentage point.
    """
    S, K, T = float(spot), float(strike), float(t)
    rd, rf, s = r_dom_pct / 100, r_for_pct / 100, vol_pct / 100
    sq = s * np.sqrt(T)
    d1 = (np.log(S / K) + (rd - rf + 0.5 * s**2) * T) / sq
    d2 = d1 - sq
    dfd, dff = np.exp(-rd * T), np.exp(-rf * T)
    if kind == "call":
        price = S * dff * norm.cdf(d1) - K * dfd * norm.cdf(d2)
        delta = dff * norm.cdf(d1)
        theta = (-S * dff * norm.pdf(d1) * s / (2 * np.sqrt(T))
                 + rf * S * dff * norm.cdf(d1) - rd * K * dfd * norm.cdf(d2))
        rho_d = K * T * dfd * norm.cdf(d2)
        rho_f = -S * T * dff * norm.cdf(d1)
    elif kind == "put":
        price = K * dfd * norm.cdf(-d2) - S * dff * norm.cdf(-d1)
        delta = -dff * norm.cdf(-d1)
        theta = (-S * dff * norm.pdf(d1) * s / (2 * np.sqrt(T))
                 - rf * S * dff * norm.cdf(-d1) + rd * K * dfd * norm.cdf(-d2))
        rho_d = -K * T * dfd * norm.cdf(-d2)
        rho_f = S * T * dff * norm.cdf(-d1)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    gamma = dff * norm.pdf(d1) / (S * sq)
    vega = S * dff * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100,
            "theta": theta / 365, "rho_dom": rho_d / 100, "rho_for": rho_f / 100,
            "d1": d1, "d2": d2, "forward": S * dff / dfd}


def implied_vol(price, spot, strike, t, r_dom_pct, r_for_pct, kind="call") -> float:
    """Volatility (%) that reproduces an option price."""
    f = lambda v: garman_kohlhagen(spot, strike, t, r_dom_pct, r_for_pct, v, kind)["price"] - price
    return brentq(f, 1e-4, 300.0, xtol=1e-10)


def ewma_vol(log_returns: pd.Series, lam: float = 0.94, periods_per_year: int = 252) -> float:
    """RiskMetrics EWMA volatility (%, annualised) at the end of the sample."""
    r = log_returns.dropna().to_numpy()
    var = r[:30].var()
    for x in r[30:]:
        var = lam * var + (1 - lam) * x**2
    return float(np.sqrt(var * periods_per_year) * 100)
