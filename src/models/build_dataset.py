# src/models/build_dataset.py
"""
Build modeling datasets.

Short = next-day target (y_day1)
Long  = horizon_days (default 63 trading days) target (y_h)
"""
from __future__ import annotations
import os
import logging
import argparse
from typing import Optional, Tuple

import pandas as pd
from src.config.logging_config import logger
from sqlalchemy import text
from src.data import db_utils



# -----------------------
# Helpers
# -----------------------
def get_engine():
    """Return SQLAlchemy engine using db_utils."""
    logger.info("Using engine from db_utils.get_engine()")
    return db_utils.get_engine()

def load_features(ticker: str = "TCS.NS") -> pd.DataFrame:

    logger.info("Loading features_daily ...")
    engine = get_engine()

    sql = text("""
        SELECT * FROM features_daily ORDER BY date ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    if df.empty:
        logger.warning("No rows returned from features_daily")
    else:
        logger.info("Loaded %d rows from features_daily.", len(df))

    # ensure date type is correct
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    df = df.sort_values("date").reset_index(drop=True)
    return df


# -----------------------
# Dataset builders
# -----------------------
def add_lag_features(df: pd.DataFrame, col: str = "return", n_lags: int = 5) -> pd.DataFrame:
    """Add lag features return_lag1 .. return_lagN."""
    df = df.copy()
    for l in range(1, n_lags + 1):
        df[f"{col}_lag{l}"] = df[col].shift(l)
    return df

def build_short_dataset(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Build dataset with next-day return target.
    - target column: y_day1 (float)
    - features: existing features + return_lag1..5
    """
    df = df_features.copy()
    if "close" not in df.columns:
        raise ValueError("features_daily must contain 'close' column")
    # compute next-day return
    df["close_next"] = df["close"].shift(-1)
    df["y_day1"] = df["close_next"] / df["close"] - 1.0
    df = add_lag_features(df, col="return", n_lags=5)
    # drop rows where target is null (last row)
    df_out = df.dropna(subset=["y_day1"]).reset_index(drop=True)
    # keep a reasonable column order
    keep_cols = ["date", "close", "return", "y_day1"] + [c for c in df_out.columns if c.startswith("return_lag")] \
                + [c for c in ["sma_7", "sma_21", "mom_7", "vol_14", "volume"] if c in df_out.columns]
    keep_cols = [c for c in keep_cols if c in df_out.columns]
    df_out = df_out[keep_cols]
    logger.info("Built short dataset rows=%d cols=%d", len(df_out), len(df_out.columns))
    return df_out

def build_long_dataset(df_features: pd.DataFrame, horizon_days: int = 63) -> pd.DataFrame:
    """
    Build long-horizon dataset.
    - target column: y_h (return over horizon_days)
    """
    df = df_features.copy()
    if "close" not in df.columns:
        raise ValueError("features_daily must contain 'close' column")
    df[f"close_t_plus_{horizon_days}"] = df["close"].shift(-horizon_days)
    df[f"y_{horizon_days}"] = df[f"close_t_plus_{horizon_days}"] / df["close"] - 1.0
    df = add_lag_features(df, col="return", n_lags=5)
    df_out = df.dropna(subset=[f"y_{horizon_days}"]).reset_index(drop=True)
    keep_cols = ["date", "close", "return", f"y_{horizon_days}"] + [c for c in df_out.columns if c.startswith("return_lag")] \
                + [c for c in ["sma_7", "sma_21", "mom_7", "vol_14", "volume"] if c in df_out.columns]
    keep_cols = [c for c in keep_cols if c in df_out.columns]
    df_out = df_out[keep_cols]
    logger.info("Built long dataset (h=%d) rows=%d cols=%d", horizon_days, len(df_out), len(df_out.columns))
    return df_out

# -----------------------
# DB write
# -----------------------
def write_dataset_to_db(df: pd.DataFrame, table_name: str):
    """Write dataset to DB replacing existing table (simple)."""
    engine = get_engine()
    logger.info("Writing %d rows to table '%s' (replace)...", len(df), table_name)
    # use to_sql (simple replace)
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    logger.info("Write complete for table '%s'.", table_name)

# -----------------------
# CLI orchestrator
# -----------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Build datasets for modeling (short & long horizons).")
    p.add_argument("cmd", choices=["show", "run", "write"], help="show=preview, run=build in-memory, write=build and write to DB")
    p.add_argument("--horizon", choices=["short", "long", "both"], default="both", help="Which dataset to build")
    p.add_argument("--horizon-days", type=int, default=63, help="Days for long horizon (default 63)")
    p.add_argument("--ticker", default=os.getenv("TICKER", "TCS.NS"), help="Ticker to filter (if features table contains ticker column)")
    return p.parse_args()

def main():
    args = _parse_args()
    cmd = args.cmd
    horizon = args.horizon
    horizon_days = args.horizon_days
    ticker = args.ticker

    df_features = load_features(ticker=ticker)
    if df_features.empty:
        logger.error("No feature rows loaded. Aborting.")
        return

    if cmd == "show":
        logger.info("SHOW: features shape=%s columns=%s", df_features.shape, df_features.columns.tolist())
        logger.info("Sample head:\n%s", df_features.head().to_string())
        return

    # build datasets as requested
    built_short = None
    built_long = None
    if horizon in ("short", "both"):
        built_short = build_short_dataset(df_features)
    if horizon in ("long", "both"):
        built_long = build_long_dataset(df_features, horizon_days=horizon_days)

    if cmd == "run":
        if built_short is not None:
            logger.info("RUN: short dataset rows=%d cols=%d", len(built_short), len(built_short.columns))
            logger.info("short sample:\n%s", built_short.tail(5).to_string(index=False))
        if built_long is not None:
            logger.info("RUN: long dataset rows=%d cols=%d", len(built_long), len(built_long.columns))
            logger.info("long sample:\n%s", built_long.tail(5).to_string(index=False))
        return

    if cmd == "write":
        if built_short is not None:
            write_dataset_to_db(built_short, "dataset_short")
        if built_long is not None:
            write_dataset_to_db(built_long, "dataset_long")
        logger.info("WRITE: done.")
        return

if __name__ == "__main__":
    main()
