"""Unit tests for the SBV interbank module: parsing, cleaning, calendar, no look-ahead."""
import numpy as np
import pandas as pd
import pytest

from src.curves import par_yields_from_curve
from src.interbank import (STRESS_WINDOW, TENORS, calendar_features, har_design, latest_interbank_curve,
                           load_interbank, parse_rate, parse_volume, stress_dataset)
from tests.test_rates import flat_curve


def test_parse_rate_and_volume():
    assert parse_rate("4,29") == pytest.approx(4.29)
    assert parse_rate("8") == 8.0
    assert np.isnan(parse_rate(""))
    assert parse_volume("804998") == (804998.0, False)
    v, ambiguous = parse_volume("432,24")          # comma in a volume: not guessed
    assert np.isnan(v) and ambiguous
    assert np.isnan(parse_volume("")[0])


def _raw_csv(path, rows):
    cols = ["date"] + [f"{t}_{k}" for t in TENORS for k in ("rate", "volume", "carried")]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_cleaning_excludes_carried_bad_and_scrambled_days(tmp_path):
    good = ["4,1", "800", "0"] * len(TENORS)
    carried = list(good); carried[-1] = "1"                          # 9M carried over
    one_bad = list(good); one_bad[3] = "512"                         # 1W impossible
    two_bad = list(good); two_bad[0] = "882"; two_bad[3] = "730"     # scrambled day
    p = tmp_path / "ib.csv"
    _raw_csv(p, [["2026-01-05"] + carried, ["2026-01-06"] + one_bad, ["2026-01-07"] + two_bad])
    df = load_interbank(p).set_index(["date", "tenor"])
    assert df.loc[("2026-01-05", "9M"), "excluded"] and not df.loc[("2026-01-05", "ON"), "excluded"]
    assert df.loc[("2026-01-06", "1W"), "excluded"] and not df.loc[("2026-01-06", "ON"), "excluded"]
    assert df.xs(pd.Timestamp("2026-01-07"), level="date")["excluded"].all()


def test_latest_curve_is_volume_weighted_over_traded_days(tmp_path):
    rows = []
    for d, rate, vol in [("2026-01-05", "4", "100"), ("2026-01-06", "6", "300"), ("2026-01-07", "9", "0")]:
        r = [rate, vol, "0"] * len(TENORS)
        rows.append([d] + r)
    rows[-1][-1] = "1"                                               # 9M on the last day: carried over
    p = tmp_path / "ib.csv"
    _raw_csv(p, rows)
    last, curve = latest_interbank_curve(load_interbank(p), days=2)
    assert last == pd.Timestamp("2026-01-07")
    assert curve["9M"] == pytest.approx(6.0)                         # only 06/01 is traded in the window
    assert curve["ON"] == pytest.approx(6.0)                         # 07/01 traded with zero volume


def test_pre_tet_window_does_not_depend_on_sample_end():
    full = pd.bdate_range("2025-12-01", "2026-03-31")
    short = full[full <= pd.Timestamp("2026-01-20")]                 # ends before Tết 2026 (17 Feb)
    cal_full, cal_short = calendar_features(full), calendar_features(short)
    assert cal_full["pre_tet"].sum() == 10
    assert cal_short["pre_tet"].sum() == 0                           # no spurious flags at the end
    assert cal_full.loc[short, "pre_tet"].equals(cal_short["pre_tet"])
    assert not (cal_full["month_end"] & cal_full["quarter_end"]).any()


@pytest.fixture(scope="module")
def overnight():
    """Synthetic overnight rate (NOT market data): AR(1) with occasional spikes."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=600)
    x = np.empty(len(idx)); x[0] = 3.0
    for i in range(1, len(idx)):
        x[i] = 3.0 + 0.9 * (x[i - 1] - 3.0) + rng.normal(0, 0.15) + (2.0 if rng.random() < 0.02 else 0.0)
    return pd.Series(x, index=idx)


def test_features_use_only_past_data(overnight):
    cut = overnight.index[400]
    full, part = stress_dataset(overnight), stress_dataset(overnight[:cut])
    feats = [c for c in full.columns if c not in ("y", "calm")]
    common = part.index[:-STRESS_WINDOW - 1]
    pd.testing.assert_frame_equal(full.loc[common, feats], part.loc[common, feats])
    h_full, h_part = har_design(overnight, 5), har_design(overnight[:cut], 5)
    cols = ["day", "week", "month"]
    pd.testing.assert_frame_equal(h_full.loc[part.index, cols], h_part[cols])


def test_stress_target_only_on_calm_days(overnight):
    D = stress_dataset(overnight)
    assert D.loc[~D["calm"], "y"].isna().all()
    assert D["y"].dropna().isin([0.0, 1.0]).all()


def test_par_yield_has_no_jump_at_fractional_maturities():
    c = flat_curve(4.0)
    par = par_yields_from_curve(c.df, np.arange(0.55, 3.0, 0.05), freq=2)
    assert np.abs(np.diff(par)).max() < 0.02                         # < 2 bp between neighbours
    assert np.allclose(par, 2 * (np.exp(0.02) - 1) * 100, atol=0.01)
