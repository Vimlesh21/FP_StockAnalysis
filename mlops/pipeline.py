"""
MLOps Pipeline Orchestrator
Manages the complete ML workflow: data ingestion, feature engineering, training, evaluation, and deployment.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

from src.config.logging_config import logger
from src.data.cleaning import run_clean_flow
from src.features.features_daily import compute_features, load_clean_daily, write_features_table
from src.data.db_utils import get_engine


class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution"""
    timestamp: str
    stage: str
    status: str
    rows_processed: int
    duration_seconds: float
    error_message: Optional[str] = None


class MLOpsPipeline:
    """
    Main MLOps pipeline orchestrator.
    Handles data, features, training, evaluation, and model deployment.
    """

    def __init__(self, ticker: str = "TCS.NS", config_path: Optional[str] = None):
        self.ticker = ticker
        self.config = self._load_config(config_path)
        self.metrics_log = []
        self.pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("mlops/runs") / self.pipeline_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Pipeline initialized: {self.pipeline_id}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load pipeline configuration"""
        default_config = {
            "data": {
                "ticker": self.ticker,
                "min_rows_required": 100,
            },
            "features": {
                "lag_periods": [1, 2, 3, 4, 5],
                "sma_periods": [7, 21],
                "momentum_period": 7,
                "volatility_period": 14,
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
                "short_horizon": 1,
                "long_horizon": 63,
            },
            "validation": {
                "min_mae_improvement": 0.01,
                "min_r2_improvement": 0.05,
            },
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                custom = json.load(f)
                default_config.update(custom)
        
        return default_config

    def _log_metric(self, stage: str, status: str, rows: int, duration: float, error: Optional[str] = None):
        """Log pipeline metrics"""
        metric = PipelineMetrics(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            status=status,
            rows_processed=rows,
            duration_seconds=duration,
            error_message=error,
        )
        self.metrics_log.append(metric)
        logger.info(f"[{stage}] Status: {status}, Rows: {rows}, Duration: {duration:.2f}s")

    def run_data_stage(self) -> Tuple[bool, Dict[str, Any]]:
        """Stage 1: Data Cleaning & QC"""
        logger.info("=" * 60)
        logger.info("STAGE 1: DATA CLEANING & QC")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        try:
            result = run_clean_flow(ticker=self.ticker, write=True)
            duration = (datetime.now() - start_time).total_seconds()
            
            daily_rows = result.get("daily_clean_rows", 0)
            hourly_rows = result.get("hourly_clean_rows", 0)
            total_rows = daily_rows + hourly_rows
            
            self._log_metric("data_cleaning", "success", total_rows, duration)
            
            return True, {
                "daily_rows": daily_rows,
                "hourly_rows": hourly_rows,
                "total_rows": total_rows,
                **result,
            }
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._log_metric("data_cleaning", "failed", 0, duration, str(e))
            logger.error(f"Data cleaning failed: {e}")
            return False, {"error": str(e)}

    def run_features_stage(self) -> Tuple[bool, Dict[str, Any]]:
        """Stage 2: Feature Engineering"""
        logger.info("=" * 60)
        logger.info("STAGE 2: FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        try:
            # Load cleaned daily data
            df_clean = load_clean_daily(self.ticker)
            if len(df_clean) < self.config["data"]["min_rows_required"]:
                raise ValueError(f"Insufficient cleaned data: {len(df_clean)} rows")
            
            # Compute features
            df_features = compute_features(df_clean)
            
            # Write to database
            write_features_table(df_features, table_name="features_daily")
            
            duration = (datetime.now() - start_time).total_seconds()
            self._log_metric("feature_engineering", "success", len(df_features), duration)
            
            return True, {
                "feature_rows": len(df_features),
                "feature_columns": list(df_features.columns),
                "date_range": f"{df_features['date'].min()} to {df_features['date'].max()}",
            }
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._log_metric("feature_engineering", "failed", 0, duration, str(e))
            logger.error(f"Feature engineering failed: {e}")
            return False, {"error": str(e)}

    def run_validation_stage(self) -> Tuple[bool, Dict[str, Any]]:
        """Stage 3: Data Validation"""
        logger.info("=" * 60)
        logger.info("STAGE 3: DATA VALIDATION")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        try:
            engine = get_engine()
            validation_results = {}
            
            # Check if features table exists and has data
            with engine.connect() as conn:
                result = conn.execute(
                    "SELECT COUNT(*) FROM features_daily WHERE date IS NOT NULL"
                )
                feature_count = result.scalar()
                validation_results["features_count"] = feature_count
                
                if feature_count < self.config["data"]["min_rows_required"]:
                    raise ValueError(f"Insufficient features: {feature_count} rows")
            
            duration = (datetime.now() - start_time).total_seconds()
            self._log_metric("validation", "success", feature_count, duration)
            
            return True, validation_results
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._log_metric("validation", "failed", 0, duration, str(e))
            logger.error(f"Validation failed: {e}")
            return False, {"error": str(e)}

    def save_metrics(self):
        """Save pipeline metrics to file"""
        metrics_file = self.run_dir / "metrics.json"
        metrics_data = [asdict(m) for m in self.metrics_log]
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

    def run_full_pipeline(self, skip_training: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the complete MLOps pipeline.
        
        Args:
            skip_training: If True, skip training stage (for prediction-only mode)
        
        Returns:
            Tuple of (success: bool, results: dict)
        """
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# MLOPS PIPELINE RUN: {self.pipeline_id}")
        logger.info(f"{'#' * 60}\n")
        
        pipeline_start = datetime.now()
        results = {
            "pipeline_id": self.pipeline_id,
            "ticker": self.ticker,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
        }
        
        # Stage 1: Data Cleaning
        success, data_result = self.run_data_stage()
        results["stages"]["data_cleaning"] = data_result
        if not success:
            self.save_metrics()
            return False, results
        
        # Stage 2: Feature Engineering
        success, features_result = self.run_features_stage()
        results["stages"]["feature_engineering"] = features_result
        if not success:
            self.save_metrics()
            return False, results
        
        # Stage 3: Validation
        success, validation_result = self.run_validation_stage()
        results["stages"]["validation"] = validation_result
        if not success:
            self.save_metrics()
            return False, results
        
        # Additional stages can be added here (training, evaluation, deployment)
        
        total_duration = (datetime.now() - pipeline_start).total_seconds()
        results["total_duration_seconds"] = total_duration
        results["status"] = "success"
        
        # Save metrics
        self.save_metrics()
        
        # Save results
        results_file = self.run_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"# Total duration: {total_duration:.2f}s")
        logger.info(f"{'#' * 60}\n")
        
        return True, results


def main():
    """CLI entry point for pipeline execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLOps Pipeline Orchestrator")
    parser.add_argument("--ticker", default="TCS.NS", help="Stock ticker symbol")
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument("--skip-training", action="store_true", help="Skip training stage")
    
    args = parser.parse_args()
    
    pipeline = MLOpsPipeline(ticker=args.ticker, config_path=args.config)
    success, results = pipeline.run_full_pipeline(skip_training=args.skip_training)
    
    if not success:
        logger.error("Pipeline failed!")
        return 1
    
    logger.info("Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
