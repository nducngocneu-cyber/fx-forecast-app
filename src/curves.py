"""Yield-curve construction: Nelson–Siegel(–Svensson) zero curves calibrated to par yields.

Published government yields (VBMA for Vietnam, FRED constant-maturity for the US) are
yields on coupon bonds trading near par, not zero-coupon rates. Discounting a cash flow
needs the zero curve, so the curve is calibrated the way a desk would do it: choose the
parameters of a smooth zero curve so that the *par yields it implies* match the published
ones. Zero rates here are continuously compounded, maturities are in years (ACT/365).

References: Nelson and Siegel (1987); Svensson (1994).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


# --------------------------------------------------------------------------- #
# Functional forms
# --------------------------------------------------------------------------- #
def ns_loadings(t, lam: float) -> np.ndarray:
    """Nelson–Siegel factor loadings (level, slope, curvature) at maturities ``t``."""
    t = np.maximum(np.asarray(t, float), 1e-8)
    x = lam * t
    slope = (1 - np.exp(-x)) / x
    curv = slope - np.exp(-x)
    return np.column_stack([np.ones_like(t), slope, curv])


def nss_zero(t, beta0, beta1, beta2, beta3, lam1, lam2) -> np.ndarray:
    """Svensson zero rate (decimal, continuous compounding). ``beta3=0`` gives Nelson–Siegel."""
    l1 = ns_loadings(t, lam1)
    t_arr = np.maximum(np.asarray(t, float), 1e-8)
    x2 = lam2 * t_arr
    curv2 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return beta0 * l1[:, 0] + beta1 * l1[:, 1] + beta2 * l1[:, 2] + beta3 * curv2


# --------------------------------------------------------------------------- #
# Curve object
# --------------------------------------------------------------------------- #
@dataclass
class ZeroCurve:
    params: dict                      # beta0..beta3, lam1, lam2 (decimals)
    model: str = "NSS"
    label: str = ""
    fit_maturities: np.ndarray = field(default_factory=lambda: np.array([]))
    fit_observed: np.ndarray = field(default_factory=lambda: np.array([]))   # % par yields
    fit_model: np.ndarray = field(default_factory=lambda: np.array([]))      # % par yields
    coupon_freq: int = 1
    bump: callable | None = None      # optional additive shift of zero rates: f(t) -> decimal

    def zero(self, t) -> np.ndarray:
        p = self.params
        z = nss_zero(t, p["beta0"], p["beta1"], p["beta2"], p["beta3"], p["lam1"], p["lam2"])
        if self.bump is not None:
            z = z + self.bump(np.asarray(t, float))
        return z

    def df(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        return np.exp(-self.zero(t) * t)

    def forward(self, t1, t2) -> np.ndarray:
        """Continuously compounded forward rate between t1 and t2 (decimal)."""
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)

    def par_yield(self, maturities, freq: int | None = None) -> np.ndarray:
        """Par yield (%) implied by the curve for bonds paying ``freq`` coupons a year."""
        return par_yields_from_curve(self.df, maturities, freq or self.coupon_freq)

    def shifted(self, shift) -> "ZeroCurve":
        """Copy whose zero rates are moved by ``shift(t)`` (decimal); used for risk and scenarios."""
        old = self.bump
        new = (lambda t: shift(t)) if old is None else (lambda t: old(t) + shift(t))
        return ZeroCurve(self.params, self.model, self.label, self.fit_maturities,
                         self.fit_observed, self.fit_model, self.coupon_freq, new)

    @property
    def rmse_bp(self) -> float:
        return float(np.sqrt(np.mean((self.fit_model - self.fit_observed) ** 2)) * 100)


class _CouponGrid:
    """All coupon times for a set of maturities in one vector, so a curve is evaluated once."""

    def __init__(self, maturities, freq: int):
        self.T = np.atleast_1d(np.asarray(maturities, float))
        self.freq = freq
        self.is_bill = self.T <= 1 / freq + 1e-9
        times, starts, accrued = [], [], []
        for T, bill in zip(self.T, self.is_bill):
            starts.append(len(times))
            if bill:
                times.append(T)
                accrued.append(0.0)
            else:
                n = int(np.ceil(T * freq - 1e-9))         # a short first period counts as a coupon date
                tt = T - np.arange(n)[::-1] / freq      # coupon dates counted back from maturity
                times.extend(tt)
                accrued.append(1 - tt[0] * freq)        # fraction of the first coupon already accrued
        self.times = np.array(times)
        self.starts = np.array(starts)
        self.accrued = np.array(accrued)
        self.last = np.r_[self.starts[1:], len(self.times)] - 1

    def par(self, df_func) -> np.ndarray:
        d = df_func(self.times)
        annuity = np.add.reduceat(d, self.starts)
        d_last = d[self.last]
        f = self.freq
        out = f * (1 - d_last) / (annuity - self.accrued)   # clean price 100 at settlement
        bill = f * (d_last ** (-1 / (f * self.T)) - 1)
        return np.where(self.is_bill, bill, out) * 100


def par_yields_from_curve(df_func, maturities, freq: int) -> np.ndarray:
    """Par coupon (%) that prices a bond at 100 under discount function ``df_func``.

    For maturities no longer than one coupon period the instrument is a zero-coupon
    bill, and the rate is quoted with ``freq`` compounding.
    """
    return _CouponGrid(maturities, freq).par(df_func)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
_LAM_GRID = [(0.3, 0.3), (0.6, 0.25), (1.0, 0.25), (1.5, 0.3), (2.5, 0.1), (0.8, 0.5)]   # (λ1, k)


def fit_curve(maturities, par_yields_pct, freq: int = 1, model: str = "NSS",
              label: str = "") -> ZeroCurve:
    """Calibrate a Nelson–Siegel(–Svensson) zero curve to observed par yields.

    Least squares on par-yield errors in basis points, several starting points for the
    decay parameters (the objective is non-convex in them), bounds that keep the curve
    economically sensible. Returns the best fit.
    """
    m = np.asarray(maturities, float)
    y = np.asarray(par_yields_pct, float)
    ok = np.isfinite(y)
    m, y = m[ok], y[ok]
    if len(m) < (6 if model == "NSS" else 4):
        model = "NS"
    level0 = y[-1] / 100
    slope0 = (y[0] - y[-1]) / 100

    # Svensson: λ2 = k·λ1 with k ≤ 0.6, so the second hump sits at longer maturities than the
    # first. Without this, λ1 ≈ λ2 makes the two curvature terms collinear and β2, β3 explode
    # in opposite directions while the fitted curve barely changes.
    def unpack(x):
        if model == "NSS":
            b0, b1, b2, b3, l1, k = x
            l2 = k * l1
        else:
            b0, b1, b2, l1 = x
            b3, l2 = 0.0, 1.0
        return dict(beta0=b0, beta1=b1, beta2=b2, beta3=b3, lam1=l1, lam2=l2)

    grid = _CouponGrid(m, freq)

    def resid(x):
        p = unpack(x)
        curve = ZeroCurve(p, model)
        return (grid.par(curve.df) - y) * 100   # bp

    if model == "NSS":
        lo = [0.0, -0.20, -0.30, -0.30, 0.10, 0.05]
        hi = [0.25, 0.20, 0.30, 0.30, 5.0, 0.60]
    else:
        lo = [0.0, -0.20, -0.30, 0.10]
        hi = [0.25, 0.20, 0.30, 5.0]

    best = None
    for l1, l2 in _LAM_GRID:
        x0 = ([level0, slope0, 0.0, 0.0, l1, l2] if model == "NSS" else [level0, slope0, 0.0, l1])
        x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        res = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale="jac")
        if best is None or res.cost < best.cost:
            best = res
    p = unpack(best.x)
    curve = ZeroCurve(p, model, label, m, y, None, freq)
    curve.fit_model = par_yields_from_curve(curve.df, m, freq)
    return curve


def key_rate_bump(tenor: float, key_tenors, size_bp: float = 1.0):
    """Triangular ("tent") zero-rate shift peaking at ``tenor`` (Ho, 1992 key-rate durations).

    The tents for neighbouring key tenors sum to a parallel shift, so key-rate DV01s add up
    to the parallel DV01.
    """
    k = np.asarray(sorted(key_tenors), float)
    i = int(np.where(np.isclose(k, tenor))[0][0])
    left = k[i - 1] if i > 0 else None
    right = k[i + 1] if i < len(k) - 1 else None

    def shift(t):
        t = np.asarray(t, float)
        w = np.zeros_like(t)
        if left is None:
            w[t <= tenor] = 1.0
        else:
            sel = (t > left) & (t <= tenor)
            w[sel] = (t[sel] - left) / (tenor - left)
        if right is None:
            w[t > tenor] = 1.0
        else:
            sel = (t > tenor) & (t < right)
            w[sel] = (right - t[sel]) / (right - tenor)
        return w * size_bp / 1e4

    return shift


def scenario_shift(points: dict[float, float]):
    """Zero-rate shift (decimal) interpolated linearly in maturity from {tenor: bp} points."""
    k = np.array(sorted(points), float)
    v = np.array([points[x] for x in sorted(points)], float)

    def shift(t):
        return np.interp(np.asarray(t, float), k, v) / 1e4

    return shift
