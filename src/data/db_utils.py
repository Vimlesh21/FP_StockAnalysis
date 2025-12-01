import os
from src.config.logging_config import logger
import pandas as pd
from sqlalchemy import create_engine, text


# ---------------------------------------------------------
# Get database connection
# ---------------------------------------------------------
def get_engine():
    """
    Create SQLAlchemy engine using DATABASE_URL.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL is not set. Please export it before running.")
        raise ValueError("DATABASE_URL not found")

    logger.info("Creating database engine...")
    engine = create_engine(db_url, future=True)
    return engine

# ---------------------------------------------------------
# Read DAILY data
# ---------------------------------------------------------
def load_daily_prices(ticker: str = "TCS.NS"):
    """
    Load entire DAILY price history for the given ticker.
    """
    logger.info(f"Loading DAILY price data for {ticker}...")

    sql = """
        SELECT date, open, high, low, close, adj_close, volume
        FROM raw_daily_prices
        WHERE ticker = %(ticker)s
        ORDER BY date ASC
    """

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"ticker": ticker})

    logger.info(f"Loaded {len(df)} daily rows for {ticker}.")
    return df


# ---------------------------------------------------------
# Read HOURLY data
# ---------------------------------------------------------
def load_hourly_prices(ticker: str = "TCS.NS"):
    """
    Load entire HOURLY price history for the given ticker.
    """
    logger.info(f"Loading HOURLY price data for {ticker}...")

    sql = """
        SELECT ts, open, high, low, close, adj_close, volume
        FROM raw_hourly_prices
        WHERE ticker = %(ticker)s
        ORDER BY ts ASC
    """

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"ticker": ticker})

    logger.info(f"Loaded {len(df)} hourly rows for {ticker}.")
    return df


# ---------------------------------------------------------
# Read latest N rows 
# ---------------------------------------------------------
def load_latest_daily(ticker="TCS.NS", n=10):
    """
    Load latest N daily rows.
    """
    logger.info(f"Loading latest {n} DAILY rows...")

    sql = f"""
        SELECT date, open, high, low, close, volume
        FROM raw_daily_prices
        WHERE ticker = :ticker
        ORDER BY date DESC
        LIMIT {n}
    """

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"ticker": ticker})

    return df


def load_latest_hourly(ticker="TCS.NS", n=10):
    """
    Load latest N hourly rows.
    """
    logger.info(f"Loading latest {n} HOURLY rows...")

    sql = f"""
        SELECT ts, open, high, low, close, volume
        FROM raw_hourly_prices
        WHERE ticker = :ticker
        ORDER BY ts DESC
        LIMIT {n}
    """

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"ticker": ticker})

    return df
