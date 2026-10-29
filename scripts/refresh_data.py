"""Refresh the bundled data snapshot and the precomputed backtest.

Usage (from the repo root):
    python -m scripts.refresh_data            # download + backtest
    python -m scripts.refresh_data --no-download

Outputs:
    data/usdvnd_daily.csv        raw daily closes from Yahoo Finance (VND=X)
    data/backtest_results.csv    one row per (model, origin, horizon)
    data/backtest_meta.json      settings used, so the app can label the results
"""
from __future__ import annotations

import argparse
import json
import time

from src.backtest import rolling_backtest
from src.config import DATA_DIR, TRAIN_WINDOWS, DEFAULT_TRAIN_WINDOW
from src.data_loader import fetch_yahoo, load_fx_data, save_snapshot

BACKTEST_SETTINGS = dict(
    train_size=TRAIN_WINDOWS[DEFAULT_TRAIN_WINDOW], horizon=30, n_origins=40, step=7, level=0.90
)


def main(download: bool = True) -> None:
    if download:
        raw = fetch_yahoo()
        save_snapshot(raw)
        print(f"Snapshot saved: {len(raw)} rows, {raw.index.min():%Y-%m-%d} → {raw.index.max():%Y-%m-%d}")

    data = load_fx_data("snapshot")
    t0 = time.time()
    bt = rolling_backtest(data.series, **BACKTEST_SETTINGS)
    bt.to_csv(DATA_DIR / "backtest_results.csv", index=False, float_format="%.4f")
    meta = {
        **BACKTEST_SETTINGS,
        "data_end": f"{data.series.index.max():%Y-%m-%d}",
        "first_origin": f"{bt['origin'].min():%Y-%m-%d}",
        "last_origin": f"{bt['origin'].max():%Y-%m-%d}",
    }
    (DATA_DIR / "backtest_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Backtest done in {time.time() - t0:.1f}s: {meta}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    main(download=not ap.parse_args().no_download)
