# src/models/evaluation_summary.py
"""
Evaluate short-term and long-term models and produce a compact summary report,
plus plots saved under `plots/`.

Usage (from repo root):
  PYTHONPATH=. python -m src.models.evaluation_summary show --ticker TCS.NS
  PYTHONPATH=. python -m src.models.evaluation_summary run  --ticker TCS.NS
  PYTHONPATH=. python -m src.models.evaluation_summary write --ticker TCS.NS
"""

from __future__ import annotations
import os
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sqlalchemy import text
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import db_utils
from src.config.logging_config import logger


# ------- helpers -------
def get_engine():
    logger.info("Using engine from db_utils.get_engine()")
    return db_utils.get_engine()

def metrics_from_arrays(y_true, y_pred) -> Dict[str, float]:
    """Return MAE, RMSE, R2 (floats)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}

def load_dataset(table_name: str) -> pd.DataFrame:
    """
    Load a dataset table (dataset_short or dataset_long).
    Returns dataframe ordered by date ascending.
    """
    engine = get_engine()
    sql = f"SELECT * FROM {table_name} ORDER BY date ASC"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info("Loaded %d rows from %s.", len(df), table_name)
    return df

def train_test_split_time_order(df: pd.DataFrame, train_fraction: float = 0.8):
    """
    Return (train_df, test_df) using first `train_fraction` fraction as train,
    preserving time order.
    """
    n = len(df)
    if n == 0:
        return df, df
    n_train = int(n * train_fraction)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    return train, test

def load_model_if_exists(path: Path):
    if not path.exists():
        logger.warning("Model file not found: %s", str(path))
        return None
    try:
        m = joblib.load(path)
        logger.info("Loaded model: %s", str(path))
        return m
    except Exception as e:
        logger.error("Failed to load model %s : %s", str(path), e)
        return None


# ------- evaluation flow -------
def evaluate_short(ticker: str = "TCS.NS") -> Dict[str, Any]:
    """
    Evaluate short-term model on dataset_short test set.
    Returns a dict with metrics and the full test arrays for plotting.
    """
    df = load_dataset("dataset_short")
    if df.empty:
        raise SystemExit("dataset_short is empty.")
    train_df, test_df = train_test_split_time_order(df)

    # Feature columns and target used in training
    feature_cols = ["return_lag1","return_lag2","return_lag3","return_lag4","return_lag5",
                    "sma_7","sma_21","mom_7","vol_14","volume"]
    target_col = "y_day1"

    # dropna the same way training did
    use_cols = [c for c in feature_cols + [target_col, "date"] if c in df.columns]
    df2 = df[use_cols].dropna()
    n_train = int(len(df2) * 0.8)
    test_df2 = df2.iloc[n_train:]

    results = {"horizon": "short", "rows_total": len(df), "rows_test": len(test_df2)}

    if len(test_df2) == 0:
        logger.warning("No test rows after dropna for short-term. Skipping.")
        return results

    X_test = test_df2[feature_cols].astype(float)
    y_test = test_df2[target_col].astype(float)

    # load model
    model_path = Path("models/short_model.pkl")
    model = load_model_if_exists(model_path)
    pred = None
    if model is not None:
        pred = model.predict(X_test)
        m = metrics_from_arrays(y_test, pred)
        results.update({"model_present": True, "model_metrics": m})
    else:
        results.update({"model_present": False})

    # baseline: yesterday (return_lag1)
    if "return_lag1" in test_df2.columns:
        baseline_pred = test_df2["return_lag1"].astype(float).to_numpy()
        baseline_m = metrics_from_arrays(y_test, baseline_pred)
    else:
        baseline_pred = [0.0] * len(y_test)
        baseline_m = metrics_from_arrays(y_test, baseline_pred)
    results.update({"baseline_metrics": baseline_m})

    # store full arrays for plotting + a small sample
    results["dates"] = test_df2["date"].astype(str).tolist()
    results["y_test_full"] = y_test.tolist()
    results["pred_model_full"] = pred.tolist() if pred is not None else None
    results["pred_baseline_full"] = baseline_pred.tolist()

    sample_out = test_df2.reset_index(drop=True).loc[:, ["date"]].copy()
    sample_out["actual"] = y_test.values
    sample_out["pred_model"] = pred.tolist() if pred is not None else None
    sample_out["pred_baseline"] = baseline_pred.tolist()
    results["sample"] = sample_out.head(15).to_dict(orient="records")

    return results

def evaluate_long(ticker: str = "TCS.NS") -> Dict[str, Any]:
    """
    Evaluate long-term model on dataset_long test set (y_63 target).
    Returns a dict similar to evaluate_short.
    """
    df = load_dataset("dataset_long")
    if df.empty:
        raise SystemExit("dataset_long is empty.")
    train_df, test_df = train_test_split_time_order(df)

    # feature/target
    feature_cols = ["return_lag1","return_lag2","return_lag3","return_lag4","return_lag5",
                    "sma_7","sma_21","mom_7","vol_14","volume"]
    target_col = "y_63"

    use_cols = [c for c in feature_cols + [target_col, "date"] if c in df.columns]
    df2 = df[use_cols].dropna()
    n_train = int(len(df2) * 0.8)
    test_df2 = df2.iloc[n_train:]

    results = {"horizon": "long", "rows_total": len(df), "rows_test": len(test_df2)}

    if len(test_df2) == 0:
        logger.warning("No test rows after dropna for long-term. Skipping.")
        return results

    X_test = test_df2[feature_cols].astype(float)
    y_test = test_df2[target_col].astype(float)

    model_path = Path("models/long_model.pkl")
    model = load_model_if_exists(model_path)
    pred = None
    if model is not None:
        pred = model.predict(X_test)
        m = metrics_from_arrays(y_test, pred)
        results.update({"model_present": True, "model_metrics": m})
    else:
        results.update({"model_present": False})

    # baseline: predict zero (common baseline for long returns)
    baseline_pred = [0.0] * len(y_test)
    baseline_m = metrics_from_arrays(y_test, baseline_pred)
    results.update({"baseline_metrics": baseline_m})

    results["dates"] = test_df2["date"].astype(str).tolist()
    results["y_test_full"] = y_test.tolist()
    results["pred_model_full"] = pred.tolist() if pred is not None else None
    results["pred_baseline_full"] = baseline_pred

    sample_out = test_df2.reset_index(drop=True).loc[:, ["date"]].copy()
    sample_out["actual"] = y_test.values
    sample_out["pred_model"] = pred.tolist() if pred is not None else None
    sample_out["pred_baseline"] = baseline_pred
    results["sample"] = sample_out.head(15).to_dict(orient="records")

    return results

def build_summary(ticker: str = "TCS.NS") -> pd.DataFrame:
    """Compute both short & long evaluation results and return a tidy DataFrame summary."""
    short_res = evaluate_short(ticker)
    long_res = evaluate_long(ticker)

    rows = []
    for res in (short_res, long_res):
        if "rows_test" not in res:
            continue
        base = {
            "horizon": res.get("horizon"),
            "rows_total": res.get("rows_total"),
            "rows_test": res.get("rows_test"),
            "model_present": res.get("model_present", False)
        }
        baseline = res.get("baseline_metrics", {})
        modelm = res.get("model_metrics", {}) if res.get("model_present") else {}
        base.update({
            "baseline_mae": baseline.get("mae"),
            "baseline_rmse": baseline.get("rmse"),
            "baseline_r2": baseline.get("r2"),
            "model_mae": modelm.get("mae"),
            "model_rmse": modelm.get("rmse"),
            "model_r2": modelm.get("r2"),
        })
        rows.append(base)

    df_out = pd.DataFrame(rows)
    return df_out

def save_reports(df_summary: pd.DataFrame, short_res: Dict, long_res: Dict, out_dir: Path = Path("reports")):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "evaluation_summary.csv"
    json_path = out_dir / "evaluation_summary.json"
    df_summary.to_csv(csv_path, index=False)
    logger.info("Saved CSV summary to: %s", str(csv_path))

    big = {"summary": df_summary.to_dict(orient="records"), "short": short_res, "long": long_res}
    with open(json_path, "w") as f:
        json.dump(big, f, indent=2, default=str)
    logger.info("Saved JSON report to: %s", str(json_path))

def save_plots(short_res: Dict, long_res: Dict, out_dir: Path = Path("plots")):
    """
    Create and save plots for short and long evaluations.
    Plots created:
      - timeseries_actual_vs_pred_{horizon}.png
      - scatter_pred_vs_actual_{horizon}.png
      - residuals_hist_{horizon}.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot(hres, prefix):
        dates = hres.get("dates")
        y = hres.get("y_test_full")
        pred = hres.get("pred_model_full")
        baseline = hres.get("pred_baseline_full")

        # convert to pandas Series (index=dates) for nicer plotting
        if dates is not None and y is not None:
            idx = pd.to_datetime(dates)
            ser_y = pd.Series(y, index=idx)
        else:
            logger.warning("No full test arrays available for %s timeseries plot.", prefix)
            return

        # timeseries plot (actual vs pred vs baseline)
        plt.figure(figsize=(10, 4))
        plt.plot(ser_y.index, ser_y.values, label="actual", linewidth=1)
        if pred is not None:
            ser_pred = pd.Series(pred, index=idx)
            plt.plot(ser_pred.index, ser_pred.values, label="pred_model", linewidth=1)
        if baseline is not None:
            ser_base = pd.Series(baseline, index=idx)
            plt.plot(ser_base.index, ser_base.values, label="pred_baseline", linewidth=1, linestyle="--")
        plt.title(f"{prefix}: actual vs model vs baseline")
        plt.legend()
        plt.tight_layout()
        p1 = out_dir / f"timeseries_actual_vs_pred_{prefix}.png"
        plt.savefig(p1)
        plt.close()
        logger.info("Saved plot: %s", str(p1))

        # scatter plot pred vs actual
        if pred is not None:
            plt.figure(figsize=(5, 5))
            plt.scatter(ser_y.values, pd.Series(pred, index=idx).values, s=6)
            plt.plot([min(ser_y.values), max(ser_y.values)], [min(ser_y.values), max(ser_y.values)], color="k", linewidth=0.8)
            plt.xlabel("actual"); plt.ylabel("predicted")
            plt.title(f"{prefix}: predicted vs actual")
            plt.tight_layout()
            p2 = out_dir / f"scatter_pred_vs_actual_{prefix}.png"
            plt.savefig(p2)
            plt.close()
            logger.info("Saved plot: %s", str(p2))

            # residual histogram
            resid = pd.Series(pred, index=idx).values - ser_y.values
            plt.figure(figsize=(6, 3))
            plt.hist(resid, bins=40)
            plt.title(f"{prefix}: residuals histogram (pred - actual)")
            plt.tight_layout()
            p3 = out_dir / f"residuals_hist_{prefix}.png"
            plt.savefig(p3)
            plt.close()
            logger.info("Saved plot: %s", str(p3))

    # short
    try:
        _plot(short_res, "short")
    except Exception as e:
        logger.exception("Failed to create short plots: %s", e)

    # long
    try:
        _plot(long_res, "long")
    except Exception as e:
        logger.exception("Failed to create long plots: %s", e)


# ------- CLI -------
def _parse_args():
    p = argparse.ArgumentParser(description="Evaluation summary for short & long models.")
    p.add_argument("cmd", choices=["show", "run", "write"], help="show=print summary, run=compute+print, write=compute+print+save files+plots")
    p.add_argument("--ticker", default="TCS.NS", help="Ticker (default TCS.NS)")
    return p.parse_args()

def main():
    args = _parse_args()
    cmd = args.cmd
    ticker = args.ticker

    logger.info("Command: %s  Ticker: %s", cmd, ticker)

    # compute
    short_res = evaluate_short(ticker)
    long_res = evaluate_long(ticker)
    df_summary = build_summary(ticker)

    # print neat summary
    logger.info("EVALUATION SUMMARY:")
    if df_summary.empty:
        logger.info("No summary rows produced.")
    else:
        logger.info("\n%s", df_summary.to_string(index=False))

    # print small sample sections
    logger.info("Short-sample (first rows):")
    if "sample" in short_res:
        for r in short_res["sample"]:
            logger.info("  %s", r)
    logger.info("Long-sample (first rows):")
    if "sample" in long_res:
        for r in long_res["sample"]:
            logger.info("  %s", r)

    if cmd == "write":
        save_reports(df_summary, short_res, long_res)
        # create plots
        save_plots(short_res, long_res)
        logger.info("WRITE: reports + plots saved.")

    return df_summary

if __name__ == "__main__":
    main()
