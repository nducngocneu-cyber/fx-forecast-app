"""In-memory analytical warehouse (DuckDB) over the bundled data snapshots.

Python loads and cleans each source once, registers the cleaned tables, and every
cross-dataset aggregation (month-end alignment, spreads, rolling volatility, the UIP
check) is done in SQL. The SQL lives in ``sql/*.sql`` so it can be read, reviewed and
re-run outside the app.

Files starting with a number and containing ``CREATE ... VIEW`` define views and run at
start-up; the others are named queries shown in the app.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import pandas as pd

from src.config import ROOT
from src.data_loader import clean_spikes, load_snapshot
from src.rates_data import load_us_yields, load_vn_yields

SQL_DIR = ROOT / "sql"
MAX_ROWS = 5000

_US_COLS = {1 / 12: "y_1m", 0.25: "y_3m", 0.5: "y_6m", 1: "y_1y", 2: "y_2y", 3: "y_3y",
            5: "y_5y", 7: "y_7y", 10: "y_10y", 20: "y_20y", 30: "y_30y"}


@dataclass(frozen=True)
class NamedQuery:
    name: str
    title: str
    sql: str


def _title(sql: str, fallback: str) -> str:
    first = sql.strip().splitlines()[0]
    return first.lstrip("- ").strip() if first.startswith("--") else fallback


def load_queries() -> tuple[list[str], list[NamedQuery]]:
    views, queries = [], []
    for path in sorted(SQL_DIR.glob("*.sql")):
        text = path.read_text()
        if re.search(r"CREATE\s+(OR\s+REPLACE\s+)?VIEW", text, re.I):
            views.append(text)
        else:
            queries.append(NamedQuery(path.stem, _title(text, path.stem), text))
    return views, queries


def source_tables() -> dict[str, pd.DataFrame]:
    """Cleaned source tables, as DataFrames, keyed by the table name used in SQL."""
    raw = load_snapshot()                       # same cleaning as the FX page, on trading days only
    cleaned, flags = clean_spikes(raw)
    fx_tbl = pd.DataFrame({"date": raw.index, "close_raw": raw.to_numpy(),
                           "close_clean": cleaned.to_numpy(), "flagged": flags.to_numpy()})

    vn = load_vn_yields()
    long = vn.yields.stack().rename("yield_pct").reset_index()
    long.columns = ["month_end", "tenor_y", "yield_pct"]
    pub = vn.raw.stack().rename("published_pct").reset_index()
    pub.columns = ["month_end", "tenor_y", "published_pct"]
    vn_tbl = long.merge(pub, on=["month_end", "tenor_y"], how="left")
    vn_tbl["tenor_y"] = vn_tbl["tenor_y"].astype(float)

    us = load_us_yields()
    us_tbl = us.drop(columns="SOFR").rename(columns=lambda c: _US_COLS[c])
    us_tbl["sofr"] = us["SOFR"]
    us_tbl = us_tbl.reset_index().rename(columns={"index": "date"})

    tables = {"fx_usdvnd": fx_tbl, "vn_yields": vn_tbl, "us_treasury": us_tbl}
    try:
        from src.interbank import load_interbank
        tables["interbank"] = load_interbank()
    except (FileNotFoundError, ImportError):
        pass
    return tables


def build_warehouse(tables: dict[str, pd.DataFrame] | None = None) -> duckdb.DuckDBPyConnection:
    """Create the in-memory database, materialise the tables, create views, then lock it down.

    After loading, file-system and network access are disabled and the configuration is
    locked, so user-typed SQL in the app can only read these tables.
    """
    tables = tables or source_tables()
    con = duckdb.connect(":memory:")
    for name, df in tables.items():
        con.register(f"_{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}_df")
        con.unregister(f"_{name}_df")
    views, _ = load_queries()
    for v in views:
        if "interbank" in v and "interbank" not in tables:
            continue
        con.execute(v)
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def run_query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


_ALLOWED = re.compile(r"^\s*(--[^\n]*\n\s*)*(WITH|SELECT|FROM|SUMMARIZE|DESCRIBE|SHOW)\b", re.I)
_FORBIDDEN = re.compile(r"\b(ATTACH|COPY|EXPORT|IMPORT|INSTALL|LOAD|PRAGMA|SET|CREATE|INSERT|UPDATE|"
                        r"DELETE|DROP|ALTER|CALL)\b", re.I)


def run_user_query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """Run a read-only query typed in the app (one statement, capped at MAX_ROWS rows)."""
    stmt = sql.strip().rstrip(";")
    if ";" in stmt:
        raise ValueError("Run one statement at a time.")
    if not _ALLOWED.match(stmt) or _FORBIDDEN.search(re.sub(r"'[^']*'", "''", stmt)):
        raise ValueError("Only read-only SELECT / WITH / SUMMARIZE / DESCRIBE queries are allowed.")
    return con.execute(f"SELECT * FROM ({stmt}) LIMIT {MAX_ROWS}").df() \
        if not re.match(r"^\s*(SUMMARIZE|DESCRIBE|SHOW)\b", stmt, re.I) else con.execute(stmt).df()
