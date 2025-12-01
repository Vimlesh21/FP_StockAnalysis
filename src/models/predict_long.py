"""
Predict long-term (63-day returns) using trained long_model.pkl.

Usage:
    python -m src.models.predict_long show --ticker TCS.NS
    python -m src.models.predict_long run --ticker TCS.NS
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from src.config.logging_config import logger

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
FEATURE_COLS = [
    "return_lag1", "return_lag2", "return_lag3",
    "return_lag4", "return_lag5",
    "sma_7", "sma_21", "mom_7", "vol_14", "volume"
]

TARGET = "y_63"

MODEL_FILE = Path("models/long_model.pkl")

PRED_OUT_DIR = Path("predictions")
PRED_OUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Load long-term test dataset (same split used in training)
# -------------------------------------------------------------------
def load_test_data(ticker: str):
    logger.info("Loading dataset_long for predictions...")

    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM dataset_long ORDER BY date ASC"), conn)

    # Keep only relevant columns
    use_cols = [c for c in FEATURE_COLS + [TARGET, "date"] if c in df.columns]
    df2 = df[use_cols].dropna()

    # Same 80/20 split
    n = len(df2)
    n_train = int(n * 0.8)
    test = df2.iloc[n_train:]

    logger.info(f"Loaded {len(test)} test rows for long-term predictions.")
    return test


# -------------------------------------------------------------------
# Run predictions
# -------------------------------------------------------------------
def run_prediction(test_df: pd.DataFrame):
    logger.info("Loading long-term model...")
    model = joblib.load(MODEL_FILE)

    X_test = test_df[FEATURE_COLS].astype(float)
    y_test = test_df[TARGET].astype(float)

    logger.info("Generating long-term predictions...")
    pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)

    logger.info(f"MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}")

    # Prepare output
    out = test_df.reset_index()[["date"]].copy()
    out["pred"] = pred
    out["actual"] = y_test.values

    return out, mae, rmse, r2


# -------------------------------------------------------------------
# Save predictions
# -------------------------------------------------------------------
def save_predictions(df: pd.DataFrame):
    out_file = PRED_OUT_DIR / "long_predictions.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"Saved long-term predictions to: {out_file}")


# -------------------------------------------------------------------
# CLI Handler
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["show", "run"])
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    test_df = load_test_data(args.ticker)
    out_df, mae, rmse, r2 = run_prediction(test_df)

    if args.mode == "show":
        print(out_df.head(15).to_string(index=False))
    else:
        save_predictions(out_df)


if __name__ == "__main__":
    main()
