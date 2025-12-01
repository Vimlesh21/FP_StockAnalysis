"""
Train / evaluate / save a short-term baseline model (predict y_day1).

Usage (examples):
  # show what data would be used (no training)
  python -m src.models.train_short show --ticker TCS.NS

  # train and evaluate (no save)
  python -m src.models.train_short run --ticker TCS.NS

  # train, evaluate and save model artifact
  python -m src.models.train_short write --ticker TCS.NS

Notes:
 - Model artifact is saved to top-level "models/short_model.pkl"
 - Uses RandomForestRegressor (simple baseline)
 - Logging is written to logs/tcsns.log
"""

from __future__ import annotations
import os
from src.config.logging_config import logger
import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text, create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------------
# Config / paths
# ---------------------
PROJECT_ROOT = Path.cwd()
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "short_model.pkl"
LOG_FILE = LOG_DIR / "tcsns.log"

# ---------------------
# DB helpers
# ---------------------
def get_engine():
    """Create SQLAlchemy engine from DATABASE_URL env var."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set in environment")
        raise SystemExit("Set DATABASE_URL env var first")
    logger.info("Creating database engine...")
    return create_engine(db_url)

def load_dataset_short(ticker: str = "TCS.NS") -> pd.DataFrame:
    """
    Load dataset_short from DB (features for short-term model).
    This expects table 'dataset_short' already created by build_dataset.
    """
    logger.info("Loading dataset_short for %s ...", ticker)
    engine = get_engine()
    sql = "SELECT * FROM dataset_short ORDER BY date ASC"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info("Loaded %d rows from dataset_short", len(df))
    return df

# ---------------------
# Prepare features / target
# ---------------------
FEATURE_COLS = [
    "return_lag1","return_lag2","return_lag3","return_lag4","return_lag5",
    "sma_7","sma_21","mom_7","vol_14","volume"
]
TARGET_COL = "y_day1"

def prepare_data(df: pd.DataFrame, require_cols=True):
    """
    Keep only feature cols + target and drop NA.
    Returns df_clean (with date leftover for reference).
    """
    cols = [c for c in FEATURE_COLS + [TARGET_COL, "date"] if c in df.columns]
    if require_cols and not set(FEATURE_COLS).issubset(set(df.columns)):
        missing = set(FEATURE_COLS) - set(df.columns)
        logger.error("Missing required feature columns: %s", missing)
        raise SystemExit(1)
    df2 = df[cols].dropna().reset_index(drop=True)
    logger.info("After dropna shape: %s", df2.shape)
    return df2

# ---------------------
# Train / evaluate / save
# ---------------------
def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators=100) -> RandomForestRegressor:
    logger.info("Training RandomForestRegressor n_estimators=%s", n_estimators)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model

def evaluate(pred, true, label="eval"):
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    r2 = r2_score(true, pred)
    logger.info("%s: MAE=%.6f, RMSE=%.6f, R2=%.6f", label, mae, rmse, r2)
    return {"mae": mae, "rmse": rmse, "r2": r2}

def save_model(model, path: Path):
    joblib.dump(model, path)
    logger.info("Saved model to %s (size=%d bytes)", path, path.stat().st_size)

# ---------------------
# CLI orchestration
# ---------------------
def run_flow(cmd: str, ticker: str = "TCS.NS"):
    # 1. load
    df = load_dataset_short(ticker)

    # 2. prepare
    df2 = prepare_data(df)

    # 3. split preserving time order (80% train)
    n = len(df2)
    n_train = int(n * 0.8)
    train = df2.iloc[:n_train]
    test = df2.iloc[n_train:]
    logger.info("Split -> train rows=%d test rows=%d", len(train), len(test))

    # X / y
    X_train = train[FEATURE_COLS].astype(float)
    y_train = train[TARGET_COL].astype(float)
    X_test = test[FEATURE_COLS].astype(float)
    y_test = test[TARGET_COL].astype(float)

    # run <= actions
    model = None
    if cmd in ("run", "write"):
        model = train_model(X_train, y_train)
        # eval train/test
        evaluate(model.predict(X_train), y_train, "train")
        evaluate(model.predict(X_test), y_test, "test")

    if cmd == "write":
        if model is None:
            logger.error("No model object to save")
            raise SystemExit(1)
        save_model(model, MODEL_FILE)
        logger.info("WRITE complete.")

    # return objects for programmatic use (useful in tests)
    return {"model": model, "train": (X_train, y_train), "test": (X_test, y_test)}

# ---------------------
# CLI
# ---------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Train short-term model.")
    p.add_argument("cmd", choices=["show", "run", "write"], help="show=inspect data, run=train+eval, write=train+eval+save")
    p.add_argument("--ticker", default=os.getenv("TICKER", "TCS.NS"))
    return p.parse_args()

def main():
    args = _parse_args()
    cmd = args.cmd
    ticker = args.ticker

    if cmd == "show":
        df = load_dataset_short(ticker)
        logger.info("SHOW: features shape=%s columns=%s", df.shape, df.columns.tolist())
        logger.info("Sample head:\n%s", df.head(10).to_string())
        return

    if cmd in ("run", "write"):
        logger.info("%s: starting", cmd.upper())
        run_flow(cmd, ticker)
        return

if __name__ == "__main__":
    main()
