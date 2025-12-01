# src/data/data_pipeline.py
from __future__ import annotations
import os
from src.config.logging_config import logger
from datetime import datetime, timedelta, date, timezone
import argparse
from typing import Optional, List, Dict, Any

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# -----------------------
# CONFIG (change if you want)
# -----------------------
DEFAULT_FALLBACK_DB = (
    "postgresql://neondb_owner:npg_pw2QhB4XGqbx@"
    "ep-muddy-union-ad4fjgcw-pooler.c-2.us-east-1.aws.neon.tech/neondb"
    "?sslmode=require&channel_binding=require"
)

TICKER = os.getenv("TICKER", "TCS.NS")
HOURLY_INTERVAL = os.getenv("HOURLY_INTERVAL", "60m")  # 60m bars by default

# -----------------------
# SQL constants
# -----------------------
UPSERT_HOURLY_SQL = text("""
    INSERT INTO raw_hourly_prices
        (ticker, ts, interval, open, high, low, close, adj_close, volume)
    VALUES
        (:ticker, :ts, :interval, :open, :high, :low, :close, :adj_close, :volume)
    ON CONFLICT (ticker, ts, interval) DO NOTHING;
""")

# -----------------------
# Helpers
# -----------------------
def get_database_url() -> str:
    url = os.getenv("DATABASE_URL") or DEFAULT_FALLBACK_DB
    if "DATABASE_URL" not in os.environ:
        logger.warning("DATABASE_URL not found in environment — using fallback URL. "
                       "Set DATABASE_URL in .env for safety.")
    return url

def get_engine(echo: bool = False) -> Engine:
    url = get_database_url()
    engine = create_engine(
        url,
        echo=echo,
        future=True,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=5,
    )
    return engine

# -----------------------
# DB setup
# -----------------------
def init_tables(engine: Optional[Engine] = None) -> None:
    """Create raw_daily_prices and raw_hourly_prices if not exist."""
    engine = engine or get_engine()
    create_daily_sql = """
    CREATE TABLE IF NOT EXISTS raw_daily_prices (
        ticker      VARCHAR(20) NOT NULL,
        date        DATE        NOT NULL,
        open        DOUBLE PRECISION,
        high        DOUBLE PRECISION,
        low         DOUBLE PRECISION,
        close       DOUBLE PRECISION,
        adj_close   DOUBLE PRECISION,
        volume      BIGINT,
        PRIMARY KEY (ticker, date)
    );
    """
    create_hourly_sql = """
    CREATE TABLE IF NOT EXISTS raw_hourly_prices (
        ticker      VARCHAR(20) NOT NULL,
        ts          TIMESTAMP   NOT NULL,
        interval    VARCHAR(10) NOT NULL,
        open        DOUBLE PRECISION,
        high        DOUBLE PRECISION,
        low         DOUBLE PRECISION,
        close       DOUBLE PRECISION,
        adj_close   DOUBLE PRECISION,
        volume      BIGINT,
        PRIMARY KEY (ticker, ts, interval)
    );
    """
    logger.info("Ensuring DB tables exist...")
    with engine.begin() as conn:
        conn.execute(text(create_daily_sql))
        conn.execute(text(create_hourly_sql))
    logger.info("✅ Tables ensured: raw_daily_prices, raw_hourly_prices")

# -----------------------
# Data fetch / ingest
# -----------------------
def backfill_daily(ticker: str = TICKER, years_back: int = 10, engine: Optional[Engine] = None) -> None:
    """
    Download N years of daily OHLCV from yfinance and insert into raw_daily_prices.
    Uses ON CONFLICT DO NOTHING.
    """
    engine = engine or get_engine()
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365 * years_back)
    logger.info(f"Backfilling daily data for {ticker} from {start_date} to {end_date} ...")

    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        logger.warning("No daily data returned by yfinance for backfill.")
        return

    # reset index -> make sure 'Date' or index become a column named 'date'
    df = df.reset_index()

    # ensure Adj Close exists
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # normalize column names, handle different pandas/yfinance formats
    df = df.rename(columns={
        "Date": "date",
        "Datetime": "date",
        "Adj Close": "adj_close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # required columns check
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    # note: we still continue even if some columns are missing, but we'll insert only present cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("Missing some expected daily columns from yfinance: %s", missing)

    # attach ticker
    df["ticker"] = ticker

    # ensure date is a plain date object (no tz)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Build df_out with columns we will insert (use get to avoid KeyError)
    df_out = df[["ticker", "date"] + [c for c in ["open","high","low","close","adj_close","volume"] if c in df.columns]].copy()

    # show sample so you can verify
    logger.info("Sample daily rows to be inserted:\n%s", df_out.head().to_string(index=False))

    insert_sql = text("""
        INSERT INTO raw_daily_prices
            (ticker, date, open, high, low, close, adj_close, volume)
        VALUES
            (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)
        ON CONFLICT (ticker, date) DO NOTHING;
    """)

    records = df_out.to_dict(orient="records")
    inserted = 0
    with engine.begin() as conn:
        for row in records:
            # skip completely invalid rows
            if not row.get("ticker") or not row.get("date"):
                logger.debug("Skipping row with missing ticker/date: %s", row)
                continue

            params = {
                "ticker": row.get("ticker"),
                "date": row.get("date"),
                "open": float(row.get("open")) if ("open" in row and pd.notna(row.get("open"))) else None,
                "high": float(row.get("high")) if ("high" in row and pd.notna(row.get("high"))) else None,
                "low": float(row.get("low")) if ("low" in row and pd.notna(row.get("low"))) else None,
                "close": float(row.get("close")) if ("close" in row and pd.notna(row.get("close"))) else None,
                "adj_close": float(row.get("adj_close")) if ("adj_close" in row and pd.notna(row.get("adj_close"))) else None,
                "volume": int(row.get("volume")) if ("volume" in row and pd.notna(row.get("volume"))) else None,
            }
            conn.execute(insert_sql, params)
            inserted += 1

    logger.info(f"✅ Backfill completed. Inserted approx {inserted} rows.")


def fetch_latest_hourly(ticker: str = TICKER, days_back: int = 600, interval: str = HOURLY_INTERVAL, engine: Optional[Engine] = None) -> None:
    """
    Fetch intraday for last `days_back` days and upsert into raw_hourly_prices.
    """
    engine = engine or get_engine()
    period = f"{days_back}d"
    logger.info(f"Downloading last {days_back} days intraday for {ticker} interval={interval} ...")
    df_hourly = yf.download(ticker, period=period, interval=interval, auto_adjust=False, prepost=False, progress=False)
    if df_hourly is None or df_hourly.empty:
        logger.warning("No hourly data returned.")
        return

    df_hourly["ts"] = df_hourly.index
    df_hourly = df_hourly.reset_index(drop=True)

    # flatten multiindex if present
    if isinstance(df_hourly.columns, pd.MultiIndex):
        df_hourly.columns = [c[0] if isinstance(c, tuple) else c for c in df_hourly.columns]

    if "Adj Close" not in df_hourly.columns and "Close" in df_hourly.columns:
        df_hourly["Adj Close"] = df_hourly["Close"]

    required_cols = ["ts", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required_cols if c not in df_hourly.columns]
    if missing:
        logger.error("Missing columns in yfinance result: %s", missing)
        return

    df_out = df_hourly[required_cols].copy()
    df_out = df_out.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    df_out["ticker"] = ticker
    df_out["interval"] = interval
    df_out["ts"] = pd.to_datetime(df_out["ts"], errors="coerce")
    df_out = df_out[df_out["ts"].notna()]

    logger.info("Sample rows to insert (hourly):\n%s", df_out.head().to_string(index=False))

    # iterate safe using records (plain dicts)
    records = df_out.to_dict(orient="records")
    attempts = 0
    with engine.begin() as conn:
        for r in records:
            params = {
                "ticker": r.get("ticker"),
                "ts": pd.Timestamp(r.get("ts")).to_pydatetime(),
                "interval": r.get("interval"),
                "open": float(r.get("open")) if pd.notna(r.get("open")) else None,
                "high": float(r.get("high")) if pd.notna(r.get("high")) else None,
                "low": float(r.get("low")) if pd.notna(r.get("low")) else None,
                "close": float(r.get("close")) if pd.notna(r.get("close")) else None,
                "adj_close": float(r.get("adj_close")) if pd.notna(r.get("adj_close")) else None,
                "volume": int(r.get("volume")) if pd.notna(r.get("volume")) else None,
            }
            conn.execute(UPSERT_HOURLY_SQL, params)
            attempts += 1

    logger.info("✅ Hourly ingestion safe upsert complete. Rows attempted: %d", attempts)

def fetch_hourly_today(ticker: str = TICKER, interval: str = HOURLY_INTERVAL, engine: Optional[Engine] = None) -> None:
    """Fetch today's intraday bars (period=1d) — safe to run multiple times."""
    engine = engine or get_engine()
    logger.info("📥 Downloading today's hourly data for %s (%s)...", ticker, interval)
    df_hourly = yf.download(ticker, period="1d", interval=interval, auto_adjust=False, prepost=False, progress=False)
    if df_hourly is None or df_hourly.empty:
        logger.warning("No hourly data returned for today.")
        return

    df_hourly["ts"] = df_hourly.index
    df_hourly = df_hourly.reset_index(drop=True)
    if isinstance(df_hourly.columns, pd.MultiIndex):
        df_hourly.columns = [c[0] if isinstance(c, tuple) else c for c in df_hourly.columns]

    if "Adj Close" not in df_hourly.columns and "Close" in df_hourly.columns:
        df_hourly["Adj Close"] = df_hourly["Close"]

    required_cols = ["ts", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required_cols if c not in df_hourly.columns]
    if missing:
        logger.error("Missing required columns for today's hourly fetch: %s", missing)
        return

    df_out = df_hourly[required_cols].copy()
    df_out = df_out.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    df_out["ticker"] = ticker
    df_out["interval"] = interval
    df_out["ts"] = pd.to_datetime(df_out["ts"], errors="coerce")
    df_out = df_out[df_out["ts"].notna()]

    logger.info("Sample cleaned hourly data:\n%s", df_out.head().to_string(index=False))

    records = df_out.to_dict(orient="records")
    inserted_attempts = 0
    with engine.begin() as conn:
        for r in records:
            params = {
                "ticker": r.get("ticker"),
                "ts": pd.Timestamp(r.get("ts")).to_pydatetime(),
                "interval": r.get("interval"),
                "open": float(r.get("open")) if pd.notna(r.get("open")) else None,
                "high": float(r.get("high")) if pd.notna(r.get("high")) else None,
                "low": float(r.get("low")) if pd.notna(r.get("low")) else None,
                "close": float(r.get("close")) if pd.notna(r.get("close")) else None,
                "adj_close": float(r.get("adj_close")) if pd.notna(r.get("adj_close")) else None,
                "volume": int(r.get("volume")) if pd.notna(r.get("volume")) else None,
            }
            conn.execute(UPSERT_HOURLY_SQL, params)
            inserted_attempts += 1

    logger.info("✅ Hourly ingestion safe upsert complete. Rows attempted: %d", inserted_attempts)

# -----------------------
# Robust hourly backfill (chunked)
# -----------------------
def backfill_hourly_years(ticker: str = TICKER, years: int = 10, interval: str = HOURLY_INTERVAL, chunk_days: int = 365, engine: Optional[Engine] = None) -> None:
    """
    Chunked hourly backfill (attempts long ranges in chunks).
    """
    engine = engine or get_engine()
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365 * years)
    logger.info("Backfilling hourly for %s from %s to %s using chunk_days=%d", ticker, start_date, end_date, chunk_days)

    curr_start = start_date
    total_attempts = 0
    while curr_start <= end_date:
        chunk_end = min(curr_start + timedelta(days=chunk_days - 1), end_date)
        logger.info("Fetching chunk: %s -> %s", curr_start.isoformat(), chunk_end.isoformat())
        try:
            df_chunk = yf.download(ticker, start=curr_start.isoformat(), end=(chunk_end + timedelta(days=1)).isoformat(), interval=interval, auto_adjust=False, prepost=False, progress=False)
        except Exception as e:
            logger.exception("yfinance chunk download failed for %s - %s: %s", curr_start, chunk_end, e)
            df_chunk = pd.DataFrame()

        if df_chunk is None or df_chunk.empty:
            logger.info("No data returned for chunk %s -> %s (may be outside provider window).", curr_start, chunk_end)
            curr_start = chunk_end + timedelta(days=1)
            continue

        # ensure index moved to ts
        if "Date" in df_chunk.columns:
            df_chunk = df_chunk.reset_index()
        else:
            df_chunk = df_chunk.reset_index()

        if isinstance(df_chunk.columns, pd.MultiIndex):
            df_chunk.columns = [c[0] if isinstance(c, tuple) else c for c in df_chunk.columns]

        if "Adj Close" not in df_chunk.columns and "Close" in df_chunk.columns:
            df_chunk["Adj Close"] = df_chunk["Close"]

        if "Datetime" in df_chunk.columns and "ts" not in df_chunk.columns:
            # sometimes the index column becomes 'Datetime' when reset
            df_chunk = df_chunk.rename(columns={"Datetime":"ts"})
        if "Date" in df_chunk.columns and "ts" not in df_chunk.columns:
            df_chunk = df_chunk.rename(columns={"Date":"ts"})

        if "ts" not in df_chunk.columns:
            logger.warning("Chunk returned unexpected columns: %s", list(df_chunk.columns))
            curr_start = chunk_end + timedelta(days=1)
            continue

        df_chunk = df_chunk.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
        df_chunk["ts"] = pd.to_datetime(df_chunk["ts"], errors="coerce")
        df_chunk = df_chunk[df_chunk["ts"].notna()]
        df_chunk["ticker"] = ticker
        df_chunk["interval"] = interval

        records = df_chunk.to_dict(orient="records")
        attempts = 0
        with engine.begin() as conn:
            for r in records:
                params = {
                    "ticker": r.get("ticker"),
                    "ts": pd.Timestamp(r.get("ts")).to_pydatetime(),
                    "interval": r.get("interval"),
                    "open": float(r.get("open")) if pd.notna(r.get("open")) else None,
                    "high": float(r.get("high")) if pd.notna(r.get("high")) else None,
                    "low": float(r.get("low")) if pd.notna(r.get("low")) else None,
                    "close": float(r.get("close")) if pd.notna(r.get("close")) else None,
                    "adj_close": float(r.get("adj_close")) if pd.notna(r.get("adj_close")) else None,
                    "volume": int(r.get("volume")) if pd.notna(r.get("volume")) else None,
                }
                conn.execute(UPSERT_HOURLY_SQL, params)
                attempts += 1

        total_attempts += attempts
        logger.info("Chunk attempted rows: %d (chunk %s->%s)", attempts, curr_start, chunk_end)
        curr_start = chunk_end + timedelta(days=1)

    logger.info("✅ Hourly backfill attempted total rows: %d", total_attempts)

# -----------------------
# Aggregation: hourly -> daily
# -----------------------
def append_daily_from_hourly(ticker: str = TICKER, engine: Optional[Engine] = None) -> None:
    """
    Aggregate hourly -> daily for dates present in raw_hourly_prices but missing in raw_daily_prices.
    """
    engine = engine or get_engine()
    with engine.begin() as conn:
        last_daily_date = conn.execute(
            text("SELECT MAX(date) FROM raw_daily_prices WHERE ticker = :ticker"),
            {"ticker": ticker},
        ).scalar()
    logger.info("Last daily date in table: %s", last_daily_date)

    with engine.begin() as conn:
        if last_daily_date is None:
            hourly_df = pd.read_sql(text("""
                SELECT ts, open, high, low, close, adj_close, volume
                FROM raw_hourly_prices
                WHERE ticker = :ticker
            """), conn, params={"ticker": ticker})
        else:
            hourly_df = pd.read_sql(text("""
                SELECT ts, open, high, low, close, adj_close, volume
                FROM raw_hourly_prices
                WHERE ticker = :ticker
                  AND ts::date > :last_date
            """), conn, params={"ticker": ticker, "last_date": last_daily_date})

    if hourly_df.empty:
        logger.info("No new hourly data beyond last daily date. Nothing to aggregate.")
        return

    hourly_df["ts"] = pd.to_datetime(hourly_df["ts"], errors="coerce")
    hourly_df = hourly_df[hourly_df["ts"].notna()]

    if hourly_df["ts"].dt.tz is not None:
        hourly_df["ts"] = hourly_df["ts"].dt.tz_convert(None)

    hourly_df["date"] = hourly_df["ts"].dt.date

    agg = hourly_df.sort_values("ts").groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        adj_close=("adj_close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    logger.info("Daily aggregates to insert:\n%s", agg.head().to_string(index=False))

    insert_sql = text("""
        INSERT INTO raw_daily_prices
            (ticker, date, open, high, low, close, adj_close, volume)
        VALUES
            (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)
        ON CONFLICT (ticker, date) DO NOTHING;
    """)
    records = agg.to_dict(orient="records")
    inserted = 0
    with engine.begin() as conn:
        for row in records:
            params = {
                "ticker": ticker,
                "date": row.get("date"),
                "open": float(row.get("open")) if pd.notna(row.get("open")) else None,
                "high": float(row.get("high")) if pd.notna(row.get("high")) else None,
                "low": float(row.get("low")) if pd.notna(row.get("low")) else None,
                "close": float(row.get("close")) if pd.notna(row.get("close")) else None,
                "adj_close": float(row.get("adj_close")) if pd.notna(row.get("adj_close")) else None,
                "volume": int(row.get("volume")) if pd.notna(row.get("volume")) else None,
            }
            conn.execute(insert_sql, params)
            inserted += 1
    logger.info("✅ Aggregated and inserted %d new daily rows from hourly data.", inserted)

# -----------------------
# Generic QC runner and checks
# -----------------------
def run_qc_checks(engine: Engine, checks: List[Dict[str, Any]], scope_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    results = []
    with engine.begin() as conn:
        for chk in checks:
            params = scope_params or {}
            row = conn.execute(text(chk["sql"]), params).fetchone()
            value = row[0] if row is not None else None
            results.append({"name": chk["name"], "level": chk.get("level", "info"), "value": value})
    return pd.DataFrame(results)

HOURLY_QC_CHECKS = [
    {
        "name": "hourly_duplicate_keys",
        "level": "error",
        "sql": """
            SELECT COUNT(*) FROM (
                SELECT ticker, ts, interval, COUNT(*) AS c
                FROM raw_hourly_prices
                WHERE ticker = :ticker
                {date_filter}
                GROUP BY ticker, ts, interval
                HAVING COUNT(*) > 1
            ) t;
        """,
    },
    {
        "name": "hourly_null_prices",
        "level": "error",
        "sql": """
            SELECT COUNT(*)
            FROM raw_hourly_prices
            WHERE ticker = :ticker
              {date_filter}
              AND (
                open IS NULL OR close IS NULL OR ts IS NULL
              );
        """,
    },
    {
        "name": "hourly_future_timestamps",
        "level": "warn",
        "sql": """
            SELECT COUNT(*)
            FROM raw_hourly_prices
            WHERE ticker = :ticker
              {date_filter}
              AND ts > NOW() + INTERVAL '5 minutes';
        """,
    },
]

def build_date_filter(column_name: str, date_from: Optional[date] = None):
    if date_from is None:
        return "", {}
    else:
        return f"AND {column_name}::date >= :date_from", {"date_from": date_from}

def qc_hourly_full(engine: Engine, ticker: str) -> pd.DataFrame:
    checks_sql = []
    for chk in HOURLY_QC_CHECKS:
        date_filter, _ = build_date_filter("ts", date_from=None)
        sql = chk["sql"].format(date_filter=date_filter)
        checks_sql.append({**chk, "sql": sql})
    df = run_qc_checks(engine, checks_sql, scope_params={"ticker": ticker})
    logger.info("Full hourly QC:\n%s", df.to_string(index=False))
    return df

def qc_hourly_incremental(engine: Engine, ticker: str, days_back: int = 1) -> pd.DataFrame:
    date_from = date.today() - timedelta(days=days_back)
    checks_sql = []
    for chk in HOURLY_QC_CHECKS:
        date_filter, _ = build_date_filter("ts", date_from=date_from)
        sql = chk["sql"].format(date_filter=date_filter)
        checks_sql.append({**chk, "sql": sql})
    params = {"ticker": ticker, "date_from": date_from}
    df = run_qc_checks(engine, checks_sql, scope_params=params)
    logger.info("Incremental hourly QC from %s:\n%s", date_from, df.to_string(index=False))
    return df

# -----------------------
# Orchestration helpers
# -----------------------
def run_daily_flow(ticker: str = TICKER, engine: Optional[Engine] = None) -> dict:
    engine = engine or get_engine()
    fetch_hourly_today(ticker=ticker, engine=engine)
    append_daily_from_hourly(ticker=ticker, engine=engine)
    qc = qc_hourly_incremental(engine=engine, ticker=ticker, days_back=1)
    return {"ticker": ticker, "qc": qc.to_dict(orient="records")}

# -----------------------
# CLI
# -----------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Data pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init_tables", help="Create required tables if not exist")
    bf = sub.add_parser("backfill_daily", help="Backfill daily history")
    bf.add_argument("--years", type=int, default=10)
    fh = sub.add_parser("fetch_latest_hourly", help="Fetch last N days of intraday")
    fh.add_argument("--days", type=int, default=600)
    sub.add_parser("fetch_hourly_today", help="Fetch today's hourly data (period=1d)")
    sub.add_parser("append_daily", help="Aggregate hourly->daily and insert new rows")
    sub.add_parser("backfill_hourly", help="Chunked hourly backfill (attempt multi-year hourly)")
    qci = sub.add_parser("qc_hourly_inc", help="Run incremental hourly QC")
    qci.add_argument("--days", type=int, default=1)
    sub.add_parser("qc_hourly_full", help="Run full hourly QC")
    sub.add_parser("run_daily_flow", help="Fetch hourly today -> append daily -> qc")

    return p.parse_args()

def main():
    args = _parse_args()
    engine = get_engine()
    cmd = args.cmd

    if cmd == "init_tables":
        init_tables(engine)
    elif cmd == "backfill_daily":
        backfill_daily(years_back=args.years, engine=engine)
    elif cmd == "fetch_latest_hourly":
        fetch_latest_hourly(days_back=args.days, engine=engine)
    elif cmd == "fetch_hourly_today":
        fetch_hourly_today(engine=engine)
    elif cmd == "append_daily":
        append_daily_from_hourly(engine=engine)
    elif cmd == "backfill_hourly":
        # default 10 years, chunk by 365 days per call
        backfill_hourly_years(engine=engine)
    elif cmd == "qc_hourly_full":
        qc_hourly_full(engine=engine, ticker=TICKER)
    elif cmd == "qc_hourly_inc":
        qc_hourly_incremental(engine=engine, ticker=TICKER, days_back=args.days)
    elif cmd == "run_daily_flow":
        res = run_daily_flow(engine=engine, ticker=TICKER)
        logger.info("run_daily_flow result: %s", res)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
