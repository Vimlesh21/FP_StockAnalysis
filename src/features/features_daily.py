"""
Feature Engineering for DAILY price data.

Commands:
 - show  : load cleaned daily table and show a sample (no feature compute)
 - run   : compute features (in-memory) and show sample (no DB write)
 - write : compute features and write to DB table 'features_daily' (replace)
"""

import os
import logging
import argparse
from typing import Tuple
from src.config.logging_config import logger
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# -----------------------
# Small DB helper
# -----------------------
def get_engine():
    """Create SQLAlchemy engine using DATABASE_URL env var."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in environment")
    return create_engine(db_url)

# -----------------------
# Load cleaned daily table
# -----------------------
def load_clean_daily(ticker: str = "TCS.NS") -> pd.DataFrame:
    """
    Load cleaned daily prices from raw_daily_prices_clean.
    Returns a DataFrame sorted by date ascending.
    """
    logger.info("Loading cleaned DAILY data for %s ...", ticker)
    sql = """
        SELECT date, open, high, low, close, adj_close, volume
        FROM raw_daily_prices_clean
        WHERE date IS NOT NULL
        ORDER BY date ASC
    """
    engine = get_engine()
    # use pandas to read
    df = pd.read_sql(text(sql), engine)
    # ensure date column is datetime and sorted
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d cleaned daily rows.", len(df))
    return df

# -----------------------
# Feature calculations (small functions)
# -----------------------
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    # simple percent change (float)
    df["return"] = df["close"].pct_change()

    close = pd.to_numeric(df["close"], errors="coerce")
    log_ret = np.log(close / close.shift(1))
    # replace inf/-inf with nan
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
    # assign back as numeric series
    df["log_return"] = pd.to_numeric(log_ret, errors="coerce")

    return df



def compute_moving_averages(df: pd.DataFrame, windows=(7, 21)) -> pd.DataFrame:
    """Add simple moving averages for given windows (sma_7, sma_21)."""
    df = df.copy()
    for w in windows:
        col = f"sma_{w}"
        df[col] = df["close"].rolling(window=w, min_periods=1).mean()
    return df

def compute_momentum_and_volatility(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    # ensure numeric
    df["log_return"] = pd.to_numeric(df.get("log_return"), errors="coerce")
    # short momentum: 7-day return
    df["mom_7"] = df["close"].pct_change(periods=7)
    # volatility: rolling std of log returns (14-day window; require at least 7)
    df["vol_14"] = df["log_return"].rolling(window=14, min_periods=7).std(ddof=0)
    # 30-day volatility example
    df["vol_30"] = df["log_return"].rolling(window=30, min_periods=15).std(ddof=0)

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper to compute all features and return new dataframe."""
    logger.info("Computing features on %d rows...", len(df))
    df2 = compute_returns(df)
    df2 = compute_moving_averages(df2, windows=(7, 21))
    df2 = compute_momentum_and_volatility(df2)

    # Keep relevant columns and ensure ordering
    keep_cols = [
        "date", "open", "high", "low", "close", "adj_close", "volume",
        "return", "log_return", "sma_7", "sma_21", "mom_7", "vol_14"
    ]
    # if some cols missing (defensive), keep intersection
    keep_cols = [c for c in keep_cols if c in df2.columns]
    df_out = df2[keep_cols].copy()
    logger.info("Features computed. Resulting columns: %s", df_out.columns.tolist())
    return df_out

# -----------------------
# Write features to DB
# -----------------------
def write_features_table(df_features: pd.DataFrame, table_name: str = "features_daily") -> None:
    """Write features DataFrame to DB replacing existing table."""
    engine = get_engine()
    # convert date to actual date type (no pandas Index tricks)
    df_out = df_features.copy()
    if "date" in df_out.columns:
        df_out["date"] = pd.to_datetime(df_out["date"]).dt.date
    logger.info("Writing %d feature rows to table '%s' (replace)...", len(df_out), table_name)
    df_out.to_sql(table_name, con=engine, if_exists="replace", index=False)
    logger.info("Write complete.")

# -----------------------
# CLI
# -----------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Daily Feature Engineering (simple)")
    p.add_argument("cmd", choices=["show", "run", "write"], help="show=load only, run=compute features (no write), write=compute+write to DB")
    p.add_argument("--ticker", default=os.getenv("TICKER", "TCS.NS"))
    return p.parse_args()

def main():
    args = _parse_args()
    cmd = args.cmd
    ticker = args.ticker

    if cmd == "show":
        logger.info("SHOW: Only loading cleaned data.")
        df_clean = load_clean_daily(ticker)
        logger.info("Shape: %s", df_clean.shape)
        logger.info("Head:\n%s", df_clean.head().to_string(index=False))
        return

    # compute features (in-memory)
    df_clean = load_clean_daily(ticker)
    df_features = compute_features(df_clean)

    if cmd == "run":
        logger.info("RUN: computed features (not written). Shape=%s", df_features.shape)
        logger.info("Sample:\n%s", df_features.tail(10).to_string(index=False))
        return

    if cmd == "write":
        logger.info("WRITE: compute and write features to DB")
        write_features_table(df_features, table_name="features_daily")
        logger.info("WRITE complete.")
        return

if __name__ == "__main__":
    main()