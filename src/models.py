"""Forecasting models with a common interface.

Every model is fitted on a price series (VND per USD) and returns, for each of
the next ``horizon`` business days, a point forecast and a central prediction
interval at the requested level:

    model = ARIMAForecaster().fit(prices)
    fc = model.predict(horizon=14, level=0.90)   # columns: point, lower, upper

All models work on log prices, so intervals are asymmetric in VND and can
never go negative. Point forecasts are medians of the predictive distribution.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BaseForecaster:
    name = "base"

    def fit(self, y: pd.Series) -> "BaseForecaster":
        if len(y) < 60:
            raise ValueError("Need at least 60 observations to fit a model.")
        self.y_ = y.astype(float)
        self.log_y_ = np.log(self.y_.to_numpy())
        self.last_date_ = y.index[-1]
        self._fit()
        return self

    def predict(self, horizon: int, level: float = 0.95) -> pd.DataFrame:
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        point, lower, upper = self._predict_log(horizon, level)
        idx = pd.bdate_range(self.last_date_ + pd.offsets.BDay(1), periods=horizon, name="date")
        return pd.DataFrame(
            {"point": np.exp(point), "lower": np.exp(lower), "upper": np.exp(upper)}, index=idx
        )

    # subclasses implement these
    def _fit(self) -> None: ...
    def _predict_log(self, horizon: int, level: float): ...

    def describe(self) -> str:
        return self.name


# --------------------------------------------------------------------------- #
# 1. Random walk – the benchmark every FX model has to beat
# --------------------------------------------------------------------------- #
class RandomWalkForecaster(BaseForecaster):
    """Naive no-change forecast: E[S_{t+h}] = S_t.

    Interval: log-returns treated as i.i.d. normal with the volatility of the
    last ``vol_window`` days, so the band widens with sqrt(h).
    """

    name = "Random walk"

    def __init__(self, vol_window: int = 250):
        self.vol_window = vol_window

    def _fit(self):
        r = np.diff(self.log_y_)[-self.vol_window:]
        self.sigma_ = float(np.std(r, ddof=1))

    def _predict_log(self, horizon, level):
        h = np.arange(1, horizon + 1)
        z = stats.norm.ppf(0.5 + level / 2)
        point = np.full(horizon, self.log_y_[-1])
        half = z * self.sigma_ * np.sqrt(h)
        return point, point - half, point + half

    def describe(self):
        return f"Random walk (σ_daily = {self.sigma_ * 1e4:.1f} bp)"


# --------------------------------------------------------------------------- #
# 2. ARIMA on log prices
# --------------------------------------------------------------------------- #
class ARIMAForecaster(BaseForecaster):
    """ARIMA(p,1,q) on log prices, with or without drift.

    If ``order`` is None the specification is chosen by AIC over
    p, q in {0..max_p} x {0..max_q} and trend in {none, drift}. In the backtest
    the order is selected once on the first training window and then held
    fixed, so model selection never sees test data.
    """

    name = "ARIMA"

    def __init__(self, order: tuple[int, int, int] | None = None, trend: str | None = None,
                 max_p: int = 2, max_q: int = 2):
        self.order = order
        self.trend = trend
        self.max_p, self.max_q = max_p, max_q

    @staticmethod
    def _fit_one(log_y, order, trend):
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on near-unit-root data
            return ARIMA(log_y, order=order, trend=trend).fit()

    def _fit(self):
        if self.order is None:
            best = (np.inf, None, None, None)
            for p, q, trend in itertools.product(range(self.max_p + 1), range(self.max_q + 1), ("n", "t")):
                try:
                    res = self._fit_one(self.log_y_, (p, 1, q), trend)
                except Exception:
                    continue
                if res.aic < best[0]:
                    best = (res.aic, (p, 1, q), trend, res)
            _, self.order_, self.trend_, self.result_ = best
            if self.result_ is None:
                raise RuntimeError("No ARIMA specification could be estimated.")
        else:
            self.order_ = self.order
            self.trend_ = self.trend or "n"
            self.result_ = self._fit_one(self.log_y_, self.order_, self.trend_)

    def _predict_log(self, horizon, level):
        fc = self.result_.get_forecast(horizon)
        ci = np.asarray(fc.conf_int(alpha=1 - level))
        return np.asarray(fc.predicted_mean), ci[:, 0], ci[:, 1]

    def describe(self):
        p, d, q = self.order_
        drift = " + drift" if self.trend_ == "t" else ""
        return f"ARIMA({p},{d},{q}){drift}, AIC = {self.result_.aic:,.0f}"


# --------------------------------------------------------------------------- #
# 3. Gradient boosting on engineered return features
# --------------------------------------------------------------------------- #
_BP = 1e4           # work in basis points so the trees see O(1) numbers
_WINDOW = 21        # prices needed to build one feature row (20 returns)
FEATURE_NAMES = [
    "r_t", "r_t-1", "r_t-2", "r_t-3", "r_t-4", "r_t-9",
    "mean_r_5d", "mean_r_20d", "vol_5d", "vol_20d", "gap_to_ma20",
]


def make_features(log_windows: np.ndarray) -> np.ndarray:
    """Features for the LAST point of each window of log prices.

    ``log_windows`` has shape (n, 21). The same function builds the training
    matrix and the recursive forecast inputs, so the two cannot drift apart.
    """
    r = np.diff(log_windows, axis=1) * _BP          # (n, 20) returns in bp
    last = log_windows[:, -1]
    ma20 = log_windows[:, -20:].mean(axis=1)
    return np.column_stack([
        r[:, -1], r[:, -2], r[:, -3], r[:, -4], r[:, -5], r[:, -10],
        r[:, -5:].mean(axis=1), r.mean(axis=1),
        r[:, -5:].std(axis=1), r.std(axis=1),
        (last - ma20) * _BP,
    ])


class GBMForecaster(BaseForecaster):
    """Histogram gradient boosting that predicts the next-day log return.

    Multi-step forecasts are built recursively: each simulated path feeds its
    own predicted return (plus a bootstrapped residual) back into the
    features. Residuals come from a hold-out split (first 80% fit, last 20%
    evaluated), because in-sample residuals of a boosted model are too small
    and would make the interval overconfident.
    """

    name = "Gradient boosting"

    def __init__(self, n_paths: int = 500, seed: int = 0):
        self.n_paths = n_paths
        self.seed = seed

    def _new_model(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_depth=3, min_samples_leaf=40,
            l2_regularization=1.0, random_state=self.seed,
        )

    def _xy(self, log_y):
        win = np.lib.stride_tricks.sliding_window_view(log_y, _WINDOW)  # win[k] ends at k+20
        X = make_features(win[:-1])
        y = np.diff(log_y)[_WINDOW - 1:] * _BP                          # return from t to t+1
        return X, y

    def _fit(self):
        X, y = self._xy(self.log_y_)
        split = int(len(y) * 0.8)
        holdout = self._new_model().fit(X[:split], y[:split])
        self.residuals_ = y[split:] - holdout.predict(X[split:])
        self.holdout_rmse_bp_ = float(np.sqrt(np.mean(self.residuals_**2)))
        self.model_ = self._new_model().fit(X, y)

    def _simulate(self, horizon):
        rng = np.random.default_rng(self.seed)
        hist = np.tile(self.log_y_[-_WINDOW:], (self.n_paths, 1))
        paths = np.empty((self.n_paths, horizon))
        for h in range(horizon):
            step = self.model_.predict(make_features(hist))
            step = step + rng.choice(self.residuals_, size=self.n_paths, replace=True)
            nxt = hist[:, -1] + step / _BP
            paths[:, h] = nxt
            hist = np.column_stack([hist[:, 1:], nxt])
        return paths

    def _predict_log(self, horizon, level):
        paths = self._simulate(horizon)
        a = (1 - level) / 2
        return (np.median(paths, axis=0), np.quantile(paths, a, axis=0),
                np.quantile(paths, 1 - a, axis=0))

    def feature_importance(self, n_repeats: int = 5) -> pd.Series:
        """Permutation importance on the training data (increase in MSE)."""
        from sklearn.inspection import permutation_importance

        X, y = self._xy(self.log_y_)
        res = permutation_importance(self.model_, X, y, n_repeats=n_repeats,
                                     random_state=self.seed, scoring="neg_mean_squared_error")
        return pd.Series(res.importances_mean, index=FEATURE_NAMES).sort_values(ascending=False)

    def describe(self):
        return f"Gradient boosting, hold-out RMSE = {self.holdout_rmse_bp_:.1f} bp/day"


MODEL_REGISTRY = {
    "Random walk": RandomWalkForecaster,
    "ARIMA": ARIMAForecaster,
    "Gradient boosting": GBMForecaster,
}
