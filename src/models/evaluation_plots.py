# src/models/evaluation_plots.py
"""
Generate simple evaluation plots for short-term and long-term predictions.

Usage (from repo root):
  PYTHONPATH=. python -m src.models.evaluation_plots run  --ticker TCS.NS
  PYTHONPATH=. python -m src.models.evaluation_plots show --ticker TCS.NS

"""

from __future__ import annotations
import argparse
from pathlib import Path
from src.config.logging_config import logger
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import db_utils


FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def safe_load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Prediction file not found: %s", path)
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def plot_timeseries(df: pd.DataFrame, horizon: str, outpath: Path, n_last: int = 200):
    """
    Plot last n_last actual vs predicted returns as lines.
    """
    df = df.sort_values("date")
    plot_df = df.tail(n_last)
    plt.figure(figsize=(10, 4.5))
    plt.plot(plot_df["date"], plot_df["actual"], label="actual", linewidth=1)
    plt.plot(plot_df["date"], plot_df["pred"], label="predicted", linewidth=1)
    plt.xlabel("date")
    plt.ylabel("return")
    plt.title(f"{horizon.upper()} — actual vs predicted (last {len(plot_df)})")
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    logger.info("Saved timeseries plot: %s", outpath)


def plot_scatter(df: pd.DataFrame, horizon: str, outpath: Path):
    """
    Scatter of actual vs predicted with y=x line and MAE in title.
    """
    plt.figure(figsize=(5, 5))
    x = df["actual"].to_numpy()
    y = df["pred"].to_numpy()
    plt.scatter(x, y, s=6, alpha=0.6)
    mn = min(np.nanmin(x), np.nanmin(y))
    mx = max(np.nanmax(x), np.nanmax(y))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=0.8)
    m = compute_metrics(x, y)
    plt.xlabel("actual")
    plt.ylabel("predicted")
    plt.title(f"{horizon.upper()} — scatter  MAE={m['mae']:.4f}")
    plt.grid(linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    logger.info("Saved scatter plot: %s", outpath)


def compute_baselines_and_metrics(short_df: pd.DataFrame | None, long_df: pd.DataFrame | None, ticker: str):
    """Compute model & baseline metrics for both horizons. Baseline: yesterday (short), zero (long)."""
    results = {}

    engine = db_utils.get_engine()

    # SHORT: need baseline = yesterday's return (return_lag1) aligned to test rows
    if short_df is not None:
        # load dataset_short to get return_lag1 aligned by date
        with engine.connect() as conn:
            ds_short = pd.read_sql("SELECT * FROM dataset_short ORDER BY date ASC", conn)
        ds_short["date"] = pd.to_datetime(ds_short["date"]).dt.date
        pred_df = short_df.copy()
        pred_df["date_only"] = pred_df["date"].dt.date
        # join on date to get return_lag1
        merged = pd.merge(pred_df, ds_short[["date", "return_lag1"]].rename(columns={"date": "date_only"}), on="date_only", how="left")
        # drop rows with missing actual or pred
        merged = merged.dropna(subset=["actual", "pred"])
        y_true = merged["actual"].astype(float).to_numpy()
        y_pred = merged["pred"].astype(float).to_numpy()
        model_metrics = compute_metrics(y_true, y_pred)
        # baseline: use return_lag1 where present else zeros
        baseline_pred = merged["return_lag1"].fillna(0.0).astype(float).to_numpy()
        baseline_metrics = compute_metrics(y_true, baseline_pred)
        results["short"] = {"model_metrics": model_metrics, "baseline_metrics": baseline_metrics, "n": len(merged)}
    else:
        results["short"] = None

    # LONG: baseline = zero
    if long_df is not None:
        df = long_df.dropna(subset=["actual", "pred"])
        y_true = df["actual"].astype(float).to_numpy()
        y_pred = df["pred"].astype(float).to_numpy()
        model_metrics = compute_metrics(y_true, y_pred)
        baseline_pred = np.zeros_like(y_true)
        baseline_metrics = compute_metrics(y_true, baseline_pred)
        results["long"] = {"model_metrics": model_metrics, "baseline_metrics": baseline_metrics, "n": len(df)}
    else:
        results["long"] = None

    return results


def make_mae_barplot(short_metrics: dict | None, long_metrics: dict | None, outpath: Path):
    """
    Create a simple bar chart comparing baseline vs model MAE for short & long.
    """
    labels = []
    baseline_vals = []
    model_vals = []
    if short_metrics:
        labels.append("short")
        baseline_vals.append(short_metrics["baseline_metrics"]["mae"])
        model_vals.append(short_metrics["model_metrics"]["mae"])
    if long_metrics:
        labels.append("long")
        baseline_vals.append(long_metrics["baseline_metrics"]["mae"])
        model_vals.append(long_metrics["model_metrics"]["mae"])

    if not labels:
        logger.warning("No metrics available for MAE barplot.")
        return

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 3.5))
    plt.bar(x - width / 2, baseline_vals, width, label="baseline")
    plt.bar(x + width / 2, model_vals, width, label="model")
    plt.xticks(x, labels)
    plt.ylabel("MAE")
    plt.title("Baseline vs Model MAE")
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    logger.info("Saved MAE comparison plot: %s", outpath)


def run_all_plots(ticker: str = "TCS.NS", save_files: bool = True):
    """Main: read prediction CSVs, compute metrics, and save plots."""
    short_path = Path("predictions/short_predictions.csv")
    long_path = Path("predictions/long_predictions.csv")

    short_df = safe_load_csv(short_path)
    long_df = safe_load_csv(long_path)

    if short_df is not None:
        # ensure columns present
        short_df = short_df.rename(columns=lambda s: s.strip())
        # some earlier outputs had no header; ensure columns exist
        if set(["date", "pred", "actual"]).issubset(short_df.columns):
            short_df["date"] = pd.to_datetime(short_df["date"])
        else:
            logger.warning("short_predictions.csv format unexpected; expected columns date,pred,actual.")
            short_df = None

    if long_df is not None:
        long_df = long_df.rename(columns=lambda s: s.strip())
        if set(["date", "pred", "actual"]).issubset(long_df.columns):
            long_df["date"] = pd.to_datetime(long_df["date"])
        else:
            logger.warning("long_predictions.csv format unexpected; expected columns date,pred,actual.")
            long_df = None

    # compute metrics & baselines
    metrics = compute_baselines_and_metrics(short_df, long_df, ticker)

    # save plots
    if short_df is not None and save_files:
        plot_timeseries(short_df, "short", FIG_DIR / "short_timeseries.png")
        plot_scatter(short_df, "short", FIG_DIR / "short_scatter.png")

    if long_df is not None and save_files:
        plot_timeseries(long_df, "long", FIG_DIR / "long_timeseries.png")
        plot_scatter(long_df, "long", FIG_DIR / "long_scatter.png")

    # mae comparison
    if save_files:
        make_mae_barplot(metrics.get("short"), metrics.get("long"), FIG_DIR / "mae_comparison.png")

    # also print a compact summary
    logger.info("PLOT SUMMARY:")
    if metrics.get("short"):
        s = metrics["short"]
        logger.info(" SHORT  n=%d  model_mae=%.6f  baseline_mae=%.6f", s["n"], s["model_metrics"]["mae"], s["baseline_metrics"]["mae"])
    else:
        logger.info(" SHORT  (no short predictions)")

    if metrics.get("long"):
        l = metrics["long"]
        logger.info(" LONG   n=%d  model_mae=%.6f  baseline_mae=%.6f", l["n"], l["model_metrics"]["mae"], l["baseline_metrics"]["mae"])
    else:
        logger.info(" LONG   (no long predictions)")

    return metrics


# ---------- CLI ----------
def _parse_args():
    p = argparse.ArgumentParser(description="Generate evaluation plots.")
    p.add_argument("cmd", choices=["show", "run"], help="show=print metrics, run=compute+save images")
    p.add_argument("--ticker", default="TCS.NS", help="Ticker")
    return p.parse_args()


def main():
    args = _parse_args()
    cmd = args.cmd
    ticker = args.ticker
    logger.info("Command: %s  Ticker: %s", cmd, ticker)

    if cmd == "show":
        metrics = run_all_plots(ticker, save_files=False)
    else:
        metrics = run_all_plots(ticker, save_files=True)

    return metrics


if __name__ == "__main__":
    main()
