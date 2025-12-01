# src/models/train_long.py
"""
Train long-term model (predict y_63).
Usage:
  python -m src.models.train_long show  --ticker TCS.NS   # show dataset info
  python -m src.models.train_long run   --ticker TCS.NS   # train, eval, save model
"""

from __future__ import annotations
import os
import argparse
import logging
from pathlib import Path
from src.config.logging_config import logger
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# use the project's db utils (keeps engine creation consistent)
from src.data import db_utils


# -------------------------
# Config
# -------------------------
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUT_DIR / "long_model.pkl"

# Features used (same base set as short-term; can be extended later)
FEATURE_COLS = [
    "return_lag1","return_lag2","return_lag3","return_lag4","return_lag5",
    "sma_7","sma_21","mom_7","vol_14","volume"
]
TARGET_COL = "y_63"

# -------------------------
# Load dataset_long from DB
# -------------------------
def load_dataset_long():
    logger.info("Loading features/dataset_long ...")
    engine = db_utils.get_engine()
    sql = "SELECT * FROM dataset_long ORDER BY date ASC"
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    logger.info("Loaded %d rows from dataset_long.", len(df))
    return df

# -------------------------
# Train / Eval helpers
# -------------------------
def split_train_test_timeordered(df: pd.DataFrame, frac_train: float = 0.8):
    n = len(df)
    n_train = int(n * frac_train)
    train = df.iloc[:n_train].copy()
    test  = df.iloc[n_train:].copy()
    return train, test

def fit_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    logger.info("Training RandomForestRegressor n_estimators=%s", n_estimators)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model

def report_metrics(y_true, y_pred, label="eval"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    logger.info("%s: MAE=%.6f, RMSE=%.6f, R2=%.6f", label, mae, rmse, r2)
    return {"mae": mae, "rmse": rmse, "r2": r2}

# -------------------------
# Main flows
# -------------------------
def show_flow(ticker: str = "TCS.NS"):
    # ticker arg kept for parity; dataset_long does not require ticker column
    df = load_dataset_long()
    logger.info("SHOW: features shape=%s columns=%s", df.shape, df.columns.tolist())
    logger.info("Sample head:\n%s", df.head().to_string())
    return df

def run_flow(ticker: str = "TCS.NS", write_model: bool = True):
    logger.info("RUN: starting")
    df = load_dataset_long()

    # keep only feature+target columns (plus date)
    cols_needed = [c for c in FEATURE_COLS + [TARGET_COL, "date"] if c in df.columns]
    df2 = df[cols_needed].dropna()
    logger.info("After dropna shape: %s", df2.shape)
    if len(df2) < 50:
        raise SystemExit("Not enough rows after dropna to train (need >=50)")

    # split
    train, test = split_train_test_timeordered(df2, frac_train=0.8)
    logger.info("Split -> train rows=%d test rows=%d", len(train), len(test))

    # prepare arrays
    X_train = train[FEATURE_COLS].astype(float)
    y_train = train[TARGET_COL].astype(float)
    X_test  = test[FEATURE_COLS].astype(float)
    y_test  = test[TARGET_COL].astype(float)

    # train
    model = fit_random_forest(X_train, y_train, n_estimators=100, random_state=42)

    # metrics
    logger.info("TRAIN metrics:")
    _ = report_metrics(y_train, model.predict(X_train), label="train")
    logger.info("TEST metrics:")
    _ = report_metrics(y_test, model.predict(X_test), label="test")

    # baseline: predict zero (no change) for long returns
    baseline_pred = np.zeros(len(y_test))
    logger.info("Baseline (predict 0) metrics on test:")
    _ = report_metrics(y_test, baseline_pred, label="baseline_zero")

    # feature importances
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": fi}).sort_values("importance", ascending=False)
        logger.info("Feature importances:\n%s", fi_df.to_string(index=False))
    else:
        logger.info("Model has no feature_importances_ attribute.")

    # save model
    if write_model:
        joblib.dump(model, MODEL_PATH)
        logger.info("Saved long model to: %s", MODEL_PATH)

    # sample predictions
    pred_test = model.predict(X_test)
    out = test.reset_index(drop=True).loc[:, ["date"]].copy()
    out["pred"] = pred_test
    out["actual"] = y_test.values
    logger.info("Sample predictions vs actual (first 10 rows):\n%s", out.head(10).to_string(index=False))

    return {"model": model, "train_rows": len(train), "test_rows": len(test)}

# -------------------------
# CLI
# -------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Train long-term model (y_63).")
    p.add_argument("cmd", choices=["show", "run"], help="show=inspect dataset_long, run=train+eval+save")
    p.add_argument("--ticker", default=os.getenv("TICKER", "TCS.NS"), help="Ticker (not required for dataset_long)")
    return p.parse_args()

def main():
    args = _parse_args()
    cmd = args.cmd
    ticker = args.ticker

    if cmd == "show":
        show_flow(ticker=ticker)
        return

    if cmd == "run":
        run_flow(ticker=ticker, write_model=True)
        return

if __name__ == "__main__":
    main()
