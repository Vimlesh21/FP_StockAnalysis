from __future__ import annotations
import os
from src.config.logging_config import logger
from datetime import datetime, date
import argparse
from typing import Optional, Dict, Any
import pandas as pd

# local project db helpers
from src.data import db_utils

# -----------------------
# Engine helper (use db_utils)
# -----------------------
def get_engine():
    """Return sqlalchemy engine from db_utils. Single place to change if needed."""
    logger.info("Using engine from db_utils.get_engine()")
    return db_utils.get_engine()

# -----------------------
# Load wrappers
# -----------------------
def load_raw_daily(ticker: str = "TCS.NS") -> pd.DataFrame:
    """
    Load raw daily prices for ticker using db_utils.load_daily_prices.
    Ensures we return a DataFrame indexed by date (datetime.date or pd.Timestamp.date).
    """
    logger.info("Loading DAILY price data for %s...", ticker)
    df = db_utils.load_daily_prices(ticker)

    # db_utils returns a DataFrame with a 'date' column
    # Convert to DatetimeIndex for cleaning convenience.
    if "date" in df.columns:
        # parse & set as index
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.set_index("date")
    else:
        # fallback: try to coerce index
        try:
            df.index = pd.to_datetime(df.index, errors="coerce").date
        except Exception:
            logger.warning("load_raw_daily: could not coerce index to dates; leaving as-is.")

    logger.info("Loaded %d daily rows for %s.", len(df), ticker)
    return df

def load_raw_hourly(ticker: str = "TCS.NS") -> pd.DataFrame:
    """
    Load raw hourly prices for ticker using db_utils.load_hourly_prices.
    Ensures we return a DataFrame indexed by timestamp (pd.Timestamp).
    """
    logger.info("Loading HOURLY price data for %s...", ticker)
    df = db_utils.load_hourly_prices(ticker)

    # db_utils returns 'ts' column; convert to timestamp index
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.set_index("ts")
    else:
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            logger.warning("load_raw_hourly: could not coerce index to timestamps; leaving as-is.")

    logger.info("Loaded %d hourly rows for %s.", len(df), ticker)
    return df

# -----------------------
# Small cleaning steps
# -----------------------
def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    - index -> datetime.date
    - lowercase columns
    - ensure 'close' exists
    - fill adj_close from close if missing
    - drop rows with missing close
    - sort ascending by date
    """
    logger.info("Cleaning daily dataframe (rows=%d)...", len(df))
    df = df.copy()

    # Coerce index to date values (pd.Timestamp -> date)
    try:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
        # convert to date (so index values are date objects)
        df.index = df.index.date
    except Exception:
        logger.warning("clean_daily: index coercion failed; continuing with original index.")

    # uniform column names
    df.columns = [str(c).lower() for c in df.columns]

    # ensure close exists
    if "close" not in df.columns:
        logger.error("clean_daily: 'close' column missing — cannot proceed.")
        return df

    # fill adj_close
    if "adj_close" in df.columns:
        df["adj_close"] = df["adj_close"].fillna(df["close"])
    else:
        df["adj_close"] = df["close"]

    # drop rows without close
    before = len(df)
    df = df[df["close"].notna()]
    dropped = before - len(df)
    if dropped:
        logger.info("clean_daily: dropped %d rows with null close", dropped)

    df = df.sort_index()
    return df

def clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    - index -> pd.Timestamp (naive)
    - drop NaT timestamps
    - drop duplicate timestamps (keep first)
    - lowercase columns
    - fill adj_close from close
    - drop rows without close
    - sort ascending
    """
    logger.info("Cleaning hourly dataframe (rows=%d)...", len(df))
    df = df.copy()

    # coerce index to datetime
    try:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
    except Exception:
        logger.warning("clean_hourly: index coercion failed; continuing with original index.")

    # drop tz (make naive) for consistent storage/aggregation
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        # if tz_localize needed
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass

    # remove duplicate timestamps
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    dup_removed = before - len(df)
    if dup_removed:
        logger.info("clean_hourly: removed %d duplicate timestamp rows", dup_removed)

    df.columns = [str(c).lower() for c in df.columns]

    # ensure close exists
    if "close" not in df.columns:
        logger.error("clean_hourly: 'close' column missing — cannot proceed.")
        return df

    # fill adj_close
    if "adj_close" in df.columns:
        df["adj_close"] = df["adj_close"].fillna(df["close"])
    else:
        df["adj_close"] = df["close"]

    # drop rows without close
    before = len(df)
    df = df[df["close"].notna()]
    dropped = before - len(df)
    if dropped:
        logger.info("clean_hourly: dropped %d rows with null close", dropped)

    df = df.sort_index()
    return df

# -----------------------
# Summaries (small)
# -----------------------
def summarize_daily(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    return {
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "nulls": df.isnull().sum().to_dict(),
    }

def summarize_hourly(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    return {
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "nulls": df.isnull().sum().to_dict(),
    }

# -----------------------
# Write cleaned tables (safe)
# -----------------------
def write_clean_tables(daily_clean: pd.DataFrame, hourly_clean: pd.DataFrame, engine=None) -> None:
    """
    Write cleaned frames to DB as:
      - raw_daily_prices_clean
      - raw_hourly_prices_clean

    This function carefully resets the index into columns named 'date' or 'ts'
    and avoids creating duplicate columns.
    """
    engine = engine or get_engine()
    logger.info("Writing cleaned tables (replace raw_*_clean)...")

    # DAILY: ensure 'date' column only once
    df_daily_out = daily_clean.copy()
    # if there is a 'date' column already (some frames might), drop to avoid duplicate column after reset
    if "date" in df_daily_out.columns:
        df_daily_out = df_daily_out.drop(columns=["date"])
    # reset index -> column will be named 'index'; rename to 'date'
    df_daily_out = df_daily_out.reset_index().rename(columns={"index": "date"})
    # ensure date column type is date for DB
    if "date" in df_daily_out.columns:
        try:
            df_daily_out["date"] = pd.to_datetime(df_daily_out["date"], errors="coerce").dt.date
        except Exception:
            pass

    # HOURLY: ensure 'ts' column only once
    df_hourly_out = hourly_clean.copy()
    if "ts" in df_hourly_out.columns:
        df_hourly_out = df_hourly_out.drop(columns=["ts"])
    df_hourly_out = df_hourly_out.reset_index().rename(columns={"index": "ts"})
    # convert ts to timestamp (ISO) for DB
    if "ts" in df_hourly_out.columns:
        try:
            df_hourly_out["ts"] = pd.to_datetime(df_hourly_out["ts"], errors="coerce")
        except Exception:
            pass

    # Write to DB (replace)
    df_daily_out.to_sql("raw_daily_prices_clean", con=engine, if_exists="replace", index=False)
    df_hourly_out.to_sql("raw_hourly_prices_clean", con=engine, if_exists="replace", index=False)

    logger.info("Wrote raw_daily_prices_clean (%d rows) and raw_hourly_prices_clean (%d rows).",
                len(df_daily_out), len(df_hourly_out))

# -----------------------
# Orchestrator
# -----------------------
def run_clean_flow(ticker: str = "TCS.NS", write: bool = False) -> Dict[str, Any]:
    """
    Load raw tables -> clean -> summarize -> optionally write cleaned tables.
    Returns summaries dict.
    """
    engine = get_engine()
    raw_daily = load_raw_daily(ticker)
    raw_hourly = load_raw_hourly(ticker)

    logger.info("Raw daily rows: %d  Raw hourly rows: %d", len(raw_daily), len(raw_hourly))

    daily_clean = clean_daily(raw_daily)
    hourly_clean = clean_hourly(raw_hourly)

    sd = summarize_daily(daily_clean)
    sh = summarize_hourly(hourly_clean)

    logger.info("DAILY: rows=%d start=%s end=%s", sd["rows"], sd.get("start"), sd.get("end"))
    logger.info("HOURLY: rows=%d start=%s end=%s dup_ts=%s", sh["rows"], sh.get("start"), sh.get("end"), sh.get("duplicate_timestamps"))

    if write:
        write_clean_tables(daily_clean, hourly_clean, engine=engine)

    return {"daily_summary": sd, "hourly_summary": sh}

# -----------------------
# CLI
# -----------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Cleaning & QC (simple, readable).")
    p.add_argument("cmd", choices=["run", "show", "write"], help="run=clean (no write), show=just show, write=clean & write cleaned tables")
    p.add_argument("--ticker", default=os.getenv("TICKER", "TCS.NS"), help="Ticker (default from env or TCS.NS)")
    return p.parse_args()

def main():
    args = _parse_args()
    cmd = args.cmd
    ticker = args.ticker

    if cmd == "show":
        logger.info("SHOW only: loading raw data and printing small sample (no cleaning write).")
        raw_daily = load_raw_daily(ticker)
        raw_hourly = load_raw_hourly(ticker)
        logger.info("Raw daily rows: %d  Raw hourly rows: %d", len(raw_daily), len(raw_hourly))
        logger.info("Daily head:\n%s", raw_daily.head().to_string())
        logger.info("Hourly head:\n%s", raw_hourly.head().to_string())
        return

    if cmd == "run":
        logger.info("RUN clean flow (no DB write).")
        res = run_clean_flow(ticker=ticker, write=False)
        logger.info("Result: %s", res)
        return

    if cmd == "write":
        logger.info("WRITE: clean flow and write tables")
        res = run_clean_flow(ticker=ticker, write=True)
        logger.info("Write completed. Result: %s", res)
        return

if __name__ == "__main__":
    main()
