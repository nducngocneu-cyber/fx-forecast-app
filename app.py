"""Treasury analytics lab – Streamlit entry point.

Run locally:  streamlit run app.py
Pages live in views/; shared settings (author, repo URL) in src/app_common.py.
"""
import streamlit as st

st.set_page_config(page_title="Treasury Analytics Lab – VND", page_icon="📈", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1250px;}
    div[data-testid="stMetricValue"] {font-size: 1.55rem;}
    .small-note {color:#6b7280; font-size:0.85rem;}
    </style>""",
    unsafe_allow_html=True,
)

pages = {
    "Forecasting": [
        st.Page("views/fx_forecast.py", title="USD/VND forecast", icon="📈", default=True),
        st.Page("views/rates_liquidity.py", title="Rates & liquidity", icon="📉"),
    ],
    "Pricing & risk": [
        st.Page("views/pricing.py", title="Curves & pricing", icon="🧮"),
        st.Page("views/portfolio.py", title="Bond portfolio", icon="⚖️"),
    ],
    "Data": [
        st.Page("views/sql_warehouse.py", title="SQL warehouse", icon="🗄️"),
    ],
}
st.navigation(pages).run()
