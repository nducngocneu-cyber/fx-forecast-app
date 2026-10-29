"""Shared Streamlit helpers: author settings, cached data/curves, footer."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.curves import fit_curve
from src.rates_data import continuous_tail, latest_curve, load_us_yields, load_vn_yields

# ---- Edit these two lines before publishing ---------------------------------
AUTHOR = "Nguyen Duc Ngoc"
REPO_URL = "https://github.com/nducngocneu-cyber/fx-forecast-app"
# -----------------------------------------------------------------------------

VN_MODEL = "NS"     # 8 published tenors: Svensson's two extra parameters hit their bounds most months
US_MODEL = "NSS"    # 11 tenors from 1M to 30Y


@st.cache_data(show_spinner=False)
def vn_history():
    h = load_vn_yields()
    return h.yields, h.raw, h.flags, continuous_tail(h.yields)


@st.cache_data(show_spinner=False)
def us_daily() -> pd.DataFrame:
    return load_us_yields()


@st.cache_resource(show_spinner="Calibrating the VND curve…")
def vn_curve(asof: pd.Timestamp):
    yields = vn_history()[0]
    row = yields.loc[asof]
    return fit_curve(row.index.to_numpy(float), row.to_numpy(), freq=1, model=VN_MODEL,
                     label=f"VND government, {asof:%b %Y}")


@st.cache_resource(show_spinner="Calibrating the USD curve…")
def us_curve(asof: pd.Timestamp | None = None):
    d, row = latest_curve(us_daily(), asof)
    c = fit_curve(row.index.to_numpy(float), row.to_numpy(), freq=2, model=US_MODEL,
                  label=f"US Treasury, {d:%d %b %Y}")
    return d, row, c


@st.cache_resource(show_spinner="Building the SQL warehouse…")
def warehouse():
    from src.warehouse import build_warehouse
    return build_warehouse()


def footer() -> None:
    text = f"Built by {AUTHOR} with Python, DuckDB, SciPy, statsmodels, scikit-learn and Streamlit."
    if "<" not in REPO_URL:
        text += f" · [Source code]({REPO_URL})"
    st.divider()
    st.markdown(f"<span class='small-note'>{text} Educational project, not investment advice.</span>",
                unsafe_allow_html=True)
