"""Vietnamese interbank money market: State Bank of Vietnam (SBV) daily averages.

Source: SBV publishes, for each business day, the average interbank rate (% a year) and
turnover (billion VND) by tenor – overnight, 1 and 2 weeks, 1, 3, 6 and 9 months – on
sbv.gov.vn ("Lãi suất thị trường liên ngân hàng"). When a tenor did not trade, SBV repeats
an earlier day's figure and marks it with a footnote; those values are flagged
``carried_over`` and excluded from statistics and models.

Liquidity analysis:
* **Calendar effects** – average overnight rate around month-ends, quarter-ends and the
  Lunar New Year (Tết), when banks' cash needs peak.
* **Overnight-rate forecast** – HAR-style regression (Corsi, 2009) on the last day, week and
  month of the rate plus calendar dummies known in advance, versus the random walk.
* **Liquidity-stress warning** – on days when the overnight rate is not already elevated,
  logistic probability that it jumps at least 100 bp above its 20-day median within the next
  5 business days, scored with the Brier skill score against the historical base rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import dm_test
from src.config import DATA_DIR

INTERBANK_PATH = DATA_DIR / "sbv_interbank_daily.csv"
TENORS = {"ON": 1, "1W": 7, "2W": 14, "1M": 30, "3M": 91, "6M": 182, "9M": 273}

# First day of the Lunar New Year (Tết Nguyên Đán).
TET = pd.to_datetime(["2014-01-31", "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16",
                      "2019-02-05", "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22",
                      "2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06"])
MAX_RATE_PCT = 30.0
STRESS_BP = 100
STRESS_WINDOW = 5


# --------------------------------------------------------------------------- #
# Parsing and loading
# --------------------------------------------------------------------------- #
def parse_rate(s) -> float:
    """Rates use a decimal comma: '4,29' -> 4.29, '8' -> 8.0, '' -> NaN."""
    s = "" if pd.isna(s) else str(s).strip()
    return float(s.replace(",", ".")) if s else np.nan


def parse_volume(s) -> tuple[float, bool]:
    """Turnover in billion VND is normally published as plain digits ('804998').

    A few values contain a comma whose meaning is ambiguous: '0,5' at 9 months is 0.5 bn,
    but '432,24' overnight in 2025 is almost certainly 432,240 bn. Those are not guessed:
    they return NaN and ambiguous=True.
    """
    s = "" if pd.isna(s) else str(s).strip()
    if not s:
        return np.nan, False
    if s.isdigit():
        return float(s), False
    return np.nan, True


def load_interbank(path=INTERBANK_PATH) -> pd.DataFrame:
    """Long table: date, tenor, tenor_days, rate_pct, volume_bn, carried_over, volume_ambiguous.

    The CSV stores SBV's figures exactly as published (decimal commas, carried-over flag per
    tenor); parsing happens here so the cleaning is visible and reproducible.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for t, days in TENORS.items():
        part = pd.DataFrame({"date": pd.to_datetime(raw["date"]), "tenor": t, "tenor_days": days})
        part["rate_pct"] = raw[f"{t}_rate"].map(parse_rate)
        vol = raw[f"{t}_volume"].map(parse_volume)
        part["volume_bn"] = [v for v, _ in vol]
        part["volume_ambiguous"] = [a for _, a in vol]
        part["carried_over"] = raw[f"{t}_carried"].eq("1")
        rows.append(part[part["rate_pct"].notna() | part["volume_bn"].notna()])
    df = pd.concat(rows, ignore_index=True).sort_values(["date", "tenor_days"]).reset_index(drop=True)

    # Data-entry errors: a rate above MAX_RATE_PCT (e.g. "882" for 8.82) is impossible; a day
    # with two or more such values is scrambled as a whole and every tenor of it is excluded.
    df["bad_value"] = df["rate_pct"] > MAX_RATE_PCT
    bad_days = df.groupby("date")["bad_value"].transform("sum") >= 2
    df["excluded"] = df["carried_over"] | df["bad_value"] | bad_days
    df["exclusion_reason"] = np.select(
        [bad_days, df["bad_value"], df["carried_over"]],
        ["scrambled day (≥2 impossible values)", f"rate above {MAX_RATE_PCT}%", "carried over (no trades)"], "")
    return df


def wide(df: pd.DataFrame, value: str = "rate_pct", traded_only: bool = True) -> pd.DataFrame:
    """Date × tenor table; by default without carried-over values and data-entry errors."""
    d = df[~df["excluded"]] if traded_only else df
    out = d.pivot_table(index="date", columns="tenor", values=value)
    return out[[t for t in TENORS if t in out.columns]].sort_index()


def latest_interbank_curve(df: pd.DataFrame | None = None, days: int = 5) -> tuple[pd.Timestamp, dict]:
    """Volume-weighted average rate per tenor over the last `days` reporting dates.

    Term tenors (6M, 9M) often trade a few hundred billion VND a day, so one day's print can be
    off the curve; weighting by volume over a week keeps the money-market curve usable for pricing.
    Only traded values count (carried-over figures and data-entry errors are excluded).
    """
    df = load_interbank() if df is None else df
    d = df[~df["excluded"]].dropna(subset=["rate_pct"])
    last = d["date"].max()
    window = sorted(d["date"].unique())[-days:]
    w = d[d["date"].isin(window)].copy()
    w["vol"] = w["volume_bn"].fillna(0.0)
    out = {}
    for t in TENORS:
        g = w[w["tenor"] == t]
        if g.empty:
            continue
        out[t] = float((g["rate_pct"] * g["vol"]).sum() / g["vol"].sum()) if g["vol"].sum() > 0 else float(g["rate_pct"].mean())
    return last, out


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Dummies for each business day: last 3 business days of a month / of a quarter,
    and the 10 business days before Tết."""
    idx = pd.DatetimeIndex(index)
    s = pd.Series(idx, index=idx)
    month = idx.to_period("M")
    rank_from_end = s.groupby(month).rank(ascending=False, method="first")
    month_end = rank_from_end <= 3
    quarter_end = month_end & idx.month.isin([3, 6, 9, 12])
    pre_tet = np.zeros(len(idx), bool)
    for t in TET:
        window = pd.bdate_range(end=t - pd.offsets.BDay(1), periods=10)   # independent of the sample's end
        pre_tet |= idx.isin(window)
    return pd.DataFrame({"month_end": month_end.to_numpy() & ~quarter_end.to_numpy(),
                         "quarter_end": quarter_end.to_numpy(), "pre_tet": pre_tet}, index=idx).astype(int)


def calendar_effects(on: pd.Series) -> pd.DataFrame:
    """Mean overnight rate in each calendar window minus the mean on other days of the same
    month (so the long-run level of rates drops out), with HAC standard errors."""
    import statsmodels.api as sm

    cal = calendar_features(on.index)
    dev = on - on.groupby(on.index.to_period("M")).transform("mean")
    X = sm.add_constant(cal)
    fit = sm.OLS(dev.to_numpy() * 100, X.to_numpy(), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    rows = []
    for i, name in enumerate(X.columns):
        if name == "const":
            continue
        rows.append({"window": name, "days": int(cal[name].sum()), "effect_bp": fit.params[i],
                     "t_stat": fit.tvalues[i], "p_value": fit.pvalues[i]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Overnight-rate forecast
# --------------------------------------------------------------------------- #
def har_design(on: pd.Series, h: int) -> pd.DataFrame:
    """Regressors known at t for forecasting the rate at t+h (calendar of the target day)."""
    cal_target = calendar_features(on.index).shift(-h)
    X = pd.DataFrame({"day": on, "week": on.rolling(5).mean(), "month": on.rolling(22).mean()},
                     index=on.index).join(cal_target.add_suffix("_target"))
    X["y"] = on.shift(-h)
    return X


def backtest_overnight(on: pd.Series, horizons=(1, 5, 10), min_train: int = 750, step: int = 5) -> pd.DataFrame:
    """Expanding-window backtest: refit every ``step`` days, forecast t+h. Returns one row per forecast."""
    rows = []
    for h in horizons:
        D = har_design(on, h).dropna(subset=["day", "week", "month"])
        cols = ["day", "week", "month", "month_end_target", "quarter_end_target", "pre_tet_target"]
        for o in range(min_train, len(D) - h, step):
            train = D.iloc[: o - h + 1].dropna()          # targets of the training rows are ≤ t
            x_now = D.iloc[o]
            if x_now[cols].isna().any() or pd.isna(x_now["y"]):
                continue
            A = np.column_stack([np.ones(len(train)), train[cols].to_numpy()])
            coef, *_ = np.linalg.lstsq(A, train["y"].to_numpy(), rcond=None)
            har = float(np.r_[1.0, x_now[cols].to_numpy(float)] @ coef)
            rows.append({"h": h, "origin": D.index[o], "actual": x_now["y"],
                         "Random walk": x_now["day"], "HAR + calendar": har})
    bt = pd.DataFrame(rows)
    return bt


def summarize_overnight(bt: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    out = []
    for h, g in bt.groupby("h"):
        e_rw = (g["actual"] - g["Random walk"]) * 100
        e_h = (g["actual"] - g["HAR + calendar"]) * 100
        _, p = dm_test((e_h**2 - e_rw**2).to_numpy(), h=h, step=step)
        out.append({"h": h, "n": len(g), "RW_RMSE_bp": np.sqrt(np.mean(e_rw**2)),
                    "HAR_RMSE_bp": np.sqrt(np.mean(e_h**2)),
                    "rel_RMSE": np.sqrt(np.mean(e_h**2)) / np.sqrt(np.mean(e_rw**2)), "DM_pvalue": p})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Liquidity-stress warning
# --------------------------------------------------------------------------- #
def stress_dataset(on: pd.Series) -> pd.DataFrame:
    """Target: max overnight rate over the next STRESS_WINDOW days ≥ 20-day median + STRESS_BP."""
    med = on.rolling(20).median()
    fwd_max = on[::-1].rolling(STRESS_WINDOW).max()[::-1].shift(-1)
    y = (fwd_max - med) * 100 >= STRESS_BP
    # The calendar is known in advance, so it is built a few days past the last observation.
    future = pd.bdate_range(on.index[-1] + pd.offsets.BDay(1), periods=STRESS_WINDOW + 1)
    cal = calendar_features(on.index.append(future))
    ahead = (cal[::-1].rolling(STRESS_WINDOW).max()[::-1].shift(-1)   # any calendar window within next 5 days
             .reindex(on.index))
    X = pd.DataFrame({
        "gap_to_median_bp": (on - med) * 100,
        "vol_20d_bp": on.diff().rolling(20).std() * 100,
        "month_end_ahead": ahead["month_end"], "quarter_end_ahead": ahead["quarter_end"],
        "pre_tet_ahead": ahead["pre_tet"],
    }, index=on.index)
    # Only days not already in stress count: predicting that an elevated rate stays elevated
    # is trivial, so the target is the *start* of a new episode.
    calm = X["gap_to_median_bp"] < STRESS_BP
    X["y"] = y.astype(float).where(fwd_max.notna() & calm)
    X["calm"] = calm
    return X


def backtest_stress(on: pd.Series, min_train: int = 750, step: int = 5) -> pd.DataFrame:
    """Out-of-sample probabilities from an expanding-window logistic regression, refit every
    ``step`` days; training rows end STRESS_WINDOW days before the origin (no look-ahead)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    D = stress_dataset(on).dropna(subset=["gap_to_median_bp", "vol_20d_bp"])
    feats = [c for c in D.columns if c not in ("y", "calm")]
    rows = []
    for o in range(min_train, len(D), step):
        train = D.iloc[: o - STRESS_WINDOW].dropna(subset=feats + ["y"])
        if train["y"].nunique() < 2:
            continue
        sc = StandardScaler().fit(train[feats])
        model = LogisticRegression(max_iter=1000).fit(sc.transform(train[feats]), train["y"])
        block = D.iloc[o: o + step].dropna(subset=feats)
        if block.empty:
            continue
        p = model.predict_proba(sc.transform(block[feats]))[:, 1]
        base = train["y"].mean()
        for d, pi, yi in zip(block.index, p, block["y"]):
            rows.append({"date": d, "p_model": pi, "p_base": base, "y": yi})
    return pd.DataFrame(rows).dropna(subset=["y"])


def brier_skill(bs: pd.DataFrame) -> dict:
    b_model = np.mean((bs["p_model"] - bs["y"]) ** 2)
    b_base = np.mean((bs["p_base"] - bs["y"]) ** 2)
    return {"brier_model": b_model, "brier_base": b_base, "skill": 1 - b_model / b_base,
            "event_rate": bs["y"].mean(), "n": len(bs)}


def fit_stress_model(on: pd.Series):
    """Fit on all labelled data; return (model, scaler, features, latest probability)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    D = stress_dataset(on).dropna(subset=["gap_to_median_bp", "vol_20d_bp"])
    feats = [c for c in D.columns if c not in ("y", "calm")]
    train = D.dropna(subset=feats + ["y"])
    sc = StandardScaler().fit(train[feats])
    model = LogisticRegression(max_iter=1000).fit(sc.transform(train[feats]), train["y"])
    latest = D[feats].dropna().iloc[[-1]]
    # Undefined when the rate is already in a stress episode (the model is for new episodes).
    p_now = float(model.predict_proba(sc.transform(latest))[0, 1]) if D["calm"].loc[latest.index[-1]] else np.nan
    coefs = pd.Series(model.coef_[0], index=feats)
    return model, sc, feats, p_now, coefs, latest.index[-1]
