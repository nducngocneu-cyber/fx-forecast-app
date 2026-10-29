"""Page: the DuckDB warehouse – tables, the named SQL queries behind the app, and a read-only console."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_common import footer, warehouse
from src.rates_plotting import AMBER, BLUE, GREY, TEAL
from src.warehouse import MAX_ROWS, load_queries, run_query, run_user_query

st.title("SQL warehouse")
st.markdown(
    "Every source is cleaned once in Python and loaded into an in-memory **DuckDB** database. Cross-dataset "
    "work – month-end alignment, spreads, rolling volatility, the UIP check – is plain SQL kept in the "
    "repository's `sql/` folder, shown here next to its result."
)

con = warehouse()
_, queries = load_queries()

tables = run_query(con, """
    SELECT table_name AS "table", estimated_size AS rows, column_count AS columns
    FROM duckdb_tables() ORDER BY table_name""")
views = run_query(con, "SELECT view_name AS view FROM duckdb_views() WHERE NOT internal ORDER BY view_name")

tab_q, tab_schema, tab_console = st.tabs(["🧾 Named queries", "🗂️ Schema", "⌨️ Console"])

CHARTS = {
    "03_vn_us_spreads": ("month_end", ["vn_minus_us_10y_bp", "vn_2s10s_bp", "us_2s10s_bp"], "basis points"),
    "04_fx_monthly": ("month_end", ["vol_3m_ann_pct"], "3-month realised vol, % a year"),
    "05_carry_vs_fx": ("month_end", ["vn_minus_us_1y_pct", "usdvnd_next_12m_pct"], "%"),
}

with tab_q:
    q = st.selectbox("Query", queries, format_func=lambda x: f"{x.name} – {x.title}")
    df = run_query(con, q.sql)
    left, right = st.columns([2, 3])
    with left:
        st.code(q.sql, language="sql")
    with right:
        if q.name in CHARTS:
            x, ys, title = CHARTS[q.name]
            long = df.melt(id_vars=x, value_vars=ys, var_name="series", value_name="value")
            fig = px.line(long, x=x, y="value", color="series", template="plotly_white",
                          color_discrete_sequence=[BLUE, AMBER, TEAL, GREY])
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=title,
                              xaxis_title=None, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0,
                                                            title=None))
            fig.update_traces(line=dict(width=2))
            st.plotly_chart(fig, width="stretch")
        st.dataframe(df, hide_index=True, width="stretch", height=300)
        st.download_button("Download result (CSV)", df.to_csv(index=False).encode(), f"{q.name}.csv", "text/csv")
    if q.name == "05_carry_vs_fx":
        stats = run_query(con, f"""SELECT corr(vn_minus_us_1y_pct, usdvnd_next_12m_pct) AS corr,
                                          regr_slope(usdvnd_next_12m_pct, vn_minus_us_1y_pct) AS slope,
                                          count(*) AS n
                                   FROM ({q.sql.strip().rstrip(';')})""").iloc[0]
        st.markdown(
            f"Uncovered interest parity predicts a slope of **+1** (USD/VND rises by the VN–US yield gap). "
            f"In this sample the regression slope is **{stats['slope']:.2f}** and the correlation "
            f"**{stats['corr']:.2f}** over {int(stats['n'])} overlapping 12-month windows – far from parity, "
            f"as in most studies of the forward-premium puzzle; the overlap makes the observations strongly "
            f"dependent, so treat it as descriptive."
        )

with tab_schema:
    c1, c2 = st.columns(2)
    c1.markdown("**Tables** (loaded from `data/`)")
    c1.dataframe(tables, hide_index=True, width="stretch")
    c2.markdown("**Views** (created from `sql/` at start-up)")
    c2.dataframe(views, hide_index=True, width="stretch")
    t = st.selectbox("Describe", list(tables["table"]) + list(views["view"]))
    st.dataframe(run_query(con, f'DESCRIBE "{t}"'), hide_index=True, width="stretch")

with tab_console:
    st.markdown(f"Read-only console: one `SELECT`/`WITH`/`SUMMARIZE`/`DESCRIBE` statement, first {MAX_ROWS:,} rows. "
                "File and network access are disabled in the database.")
    default = ("SELECT tenor_y, round(avg(yield_pct), 3) AS avg_yield, round(stddev(yield_pct), 3) AS sd\n"
               "FROM vn_yields\nWHERE month_end >= DATE '2025-01-01'\nGROUP BY tenor_y\nORDER BY tenor_y")
    sql = st.text_area("SQL", default, height=160)
    if st.button("Run query", type="primary"):
        try:
            st.dataframe(run_user_query(con, sql), hide_index=True, width="stretch")
        except Exception as e:  # syntax errors, blocked statements
            st.error(f"{type(e).__name__}: {e}")

footer()
