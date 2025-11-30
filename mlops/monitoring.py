"""
MLOps Monitoring & Alerting System
Tracks model performance, data quality, and system health.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from enum import Enum

from src.config.logging_config import logger
from src.data.db_utils import get_engine


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """System health check result"""
    timestamp: str
    database_connected: bool
    models_available: bool
    features_stale: bool
    data_quality_score: float
    alerts: List[Dict[str, str]]


class ModelMonitor:
    """Monitor model performance and data quality"""

    def __init__(self, models_dir: str = "models", reports_dir: str = "reports"):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.alerts = []

    def check_model_staleness(self, threshold_days: int = 7) -> Tuple[bool, str]:
        """Check if models are stale"""
        short_model = self.models_dir / "short_model.pkl"
        long_model = self.models_dir / "long_model.pkl"
        
        if not short_model.exists() or not long_model.exists():
            return False, "Models not found"
        
        short_mtime = datetime.fromtimestamp(short_model.stat().st_mtime)
        age_days = (datetime.now() - short_mtime).days
        
        if age_days > threshold_days:
            return True, f"Models are {age_days} days old (threshold: {threshold_days})"
        
        return False, f"Models are {age_days} days old"

    def check_database_connection(self) -> Tuple[bool, str]:
        """Check database connectivity"""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                result = conn.execute("SELECT 1")
                return True, "Database connected"
        except Exception as e:
            return False, f"Database error: {str(e)}"

    def check_feature_data_quality(self, ticker: str = "TCS.NS") -> Tuple[float, Dict[str, any]):
        """Check quality of feature data"""
        try:
            engine = get_engine()
            
            with engine.connect() as conn:
                # Get feature statistics
                query = f"""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(*) FILTER (WHERE date IS NOT NULL) as non_null_dates,
                    COUNT(*) FILTER (WHERE return_lag1 IS NOT NULL) as non_null_features,
                    MAX(date) as latest_date
                FROM features_daily
                """
                df = pd.read_sql(query, conn)
                
                if df.empty or df.iloc[0]['total_rows'] == 0:
                    return 0.0, {"status": "no_data"}
                
                row = df.iloc[0]
                completeness = (row['non_null_features'] / max(row['total_rows'], 1)) * 100
                
                # Check if data is recent (last 7 days)
                latest_date = pd.to_datetime(row['latest_date'])
                days_since_update = (datetime.now() - latest_date).days
                
                quality_score = completeness * 0.7  # 70% weight on completeness
                if days_since_update <= 1:
                    quality_score += 30  # 30% weight on recency
                else:
                    quality_score += max(0, 30 - (days_since_update * 5))
                
                return quality_score, {
                    "total_rows": int(row['total_rows']),
                    "non_null_features": int(row['non_null_features']),
                    "completeness_pct": float(completeness),
                    "latest_date": str(latest_date.date()),
                    "days_since_update": days_since_update,
                }
        except Exception as e:
            logger.error(f"Feature quality check failed: {e}")
            return 0.0, {"error": str(e)}

    def check_model_performance_drift(self) -> Tuple[float, Dict[str, any]):
        """Check for model performance degradation"""
        eval_file = self.reports_dir / "evaluation_summary.csv"
        
        if not eval_file.exists():
            return 0.0, {"status": "no_evaluation_data"}
        
        try:
            eval_df = pd.read_csv(eval_file)
            
            # Use R² score as primary metric
            short_r2 = eval_df[eval_df['horizon'] == 'short']['model_r2'].values
            
            if len(short_r2) == 0:
                return 0.0, {"status": "invalid_evaluation_data"}
            
            r2_score = float(short_r2[0])
            
            # Normalize R² to 0-100 scale
            # Assume R² < -1 is worst, R² > 0.5 is excellent
            performance_score = max(0, min(100, (r2_score + 1) * 50))
            
            return performance_score, {
                "short_model_r2": r2_score,
                "performance_level": "excellent" if r2_score > 0.3 else "good" if r2_score > 0 else "poor"
            }
        except Exception as e:
            logger.error(f"Performance drift check failed: {e}")
            return 0.0, {"error": str(e)}

    def run_health_check(self) -> HealthCheck:
        """Run complete system health check"""
        
        # Database
        db_ok, db_msg = self.check_database_connection()
        
        # Models
        model_stale, model_msg = self.check_model_staleness()
        models_ok = not model_stale and self.models_dir.joinpath("short_model.pkl").exists()
        
        # Features
        feature_quality, feature_details = self.check_feature_data_quality()
        features_stale = feature_details.get('days_since_update', 999) > 1
        
        # Performance
        perf_score, perf_details = self.check_model_performance_drift()
        
        # Aggregate alerts
        alerts = []
        
        if not db_ok:
            alerts.append({
                "severity": AlertSeverity.CRITICAL.value,
                "message": db_msg,
            })
        
        if model_stale:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "message": model_msg,
            })
        
        if features_stale:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "message": f"Feature data is stale ({feature_details.get('days_since_update')} days old)",
            })
        
        if perf_score < 20:
            alerts.append({
                "severity": AlertSeverity.CRITICAL.value,
                "message": f"Model performance degraded (score: {perf_score:.1f})",
            })
        
        # Data quality threshold
        if feature_quality < 50:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "message": f"Data quality low ({feature_quality:.1f}%)",
            })
        
        health_check = HealthCheck(
            timestamp=datetime.now().isoformat(),
            database_connected=db_ok,
            models_available=models_ok,
            features_stale=features_stale,
            data_quality_score=feature_quality,
            alerts=alerts,
        )
        
        return health_check

    def save_health_report(self, health_check: HealthCheck, output_dir: str = "mlops/health"):
        """Save health check report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(asdict(health_check), f, indent=2)
        
        logger.info(f"Health report saved to {report_file}")
        return report_file


class MetricsCollector:
    """Collect and aggregate metrics over time"""

    def __init__(self, metrics_dir: str = "mlops/metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def save_metric(self, metric_name: str, metric_value: float, tags: Optional[Dict] = None):
        """Save a metric point"""
        timestamp = datetime.now()
        
        metric_data = {
            "timestamp": timestamp.isoformat(),
            "name": metric_name,
            "value": metric_value,
            "tags": tags or {},
        }
        
        # Append to metric file (time-series style)
        metric_file = self.metrics_dir / f"{metric_name}.jsonl"
        with open(metric_file, 'a') as f:
            f.write(json.dumps(metric_data) + '\n')

    def get_metric_history(self, metric_name: str, days: int = 7) -> pd.DataFrame:
        """Get historical metrics"""
        metric_file = self.metrics_dir / f"{metric_name}.jsonl"
        
        if not metric_file.exists():
            return pd.DataFrame()
        
        data = []
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with open(metric_file) as f:
            for line in f:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry['timestamp'])
                if timestamp > cutoff_time:
                    data.append(entry)
        
        return pd.DataFrame(data)

    def generate_metrics_report(self) -> Dict[str, any]:
        """Generate comprehensive metrics report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
        }
        
        for metric_file in self.metrics_dir.glob("*.jsonl"):
            metric_name = metric_file.stem
            df = self.get_metric_history(metric_name, days=30)
            
            if len(df) > 0:
                report["metrics"][metric_name] = {
                    "latest_value": float(df.iloc[-1]['value']),
                    "min": float(df['value'].min()),
                    "max": float(df['value'].max()),
                    "mean": float(df['value'].mean()),
                    "std": float(df['value'].std()),
                    "trend": "up" if df.iloc[-1]['value'] > df.iloc[0]['value'] else "down",
                }
        
        return report


class DeploymentManager:
    """Manage model deployment and versioning"""

    def __init__(self, models_dir: str = "models", archive_dir: str = "mlops/model_archive"):
        self.models_dir = Path(models_dir)
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def version_models(self, version_tag: str):
        """Create versioned backup of current models"""
        version_dir = self.archive_dir / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)
        
        short_model = self.models_dir / "short_model.pkl"
        long_model = self.models_dir / "long_model.pkl"
        
        if short_model.exists():
            import shutil
            shutil.copy(short_model, version_dir / "short_model.pkl")
            logger.info(f"Versioned short model: {version_dir}/short_model.pkl")
        
        if long_model.exists():
            import shutil
            shutil.copy(long_model, version_dir / "long_model.pkl")
            logger.info(f"Versioned long model: {version_dir}/long_model.pkl")
        
        # Save metadata
        metadata = {
            "version_tag": version_tag,
            "timestamp": datetime.now().isoformat(),
            "models": {
                "short_model": short_model.exists(),
                "long_model": long_model.exists(),
            }
        }
        
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return version_dir

    def rollback_to_version(self, version_tag: str) -> bool:
        """Rollback to a previous model version"""
        version_dir = self.archive_dir / version_tag
        
        if not version_dir.exists():
            logger.error(f"Version {version_tag} not found")
            return False
        
        try:
            import shutil
            short_src = version_dir / "short_model.pkl"
            long_src = version_dir / "long_model.pkl"
            
            if short_src.exists():
                shutil.copy(short_src, self.models_dir / "short_model.pkl")
            if long_src.exists():
                shutil.copy(long_src, self.models_dir / "long_model.pkl")
            
            logger.info(f"Rolled back to version {version_tag}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def list_versions(self) -> List[str]:
        """List all available model versions"""
        return sorted([d.name for d in self.archive_dir.iterdir() if d.is_dir()], reverse=True)


def main():
    """CLI for monitoring and health checks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLOps Monitoring & Health Checks")
    parser.add_argument("command", choices=["health", "metrics", "version"],
                       help="Command to run")
    parser.add_argument("--version-tag", help="Version tag for versioning")
    
    args = parser.parse_args()
    
    if args.command == "health":
        monitor = ModelMonitor()
        health = monitor.run_health_check()
        monitor.save_health_report(health)
        print(json.dumps(asdict(health), indent=2))
    
    elif args.command == "metrics":
        collector = MetricsCollector()
        report = collector.generate_metrics_report()
        print(json.dumps(report, indent=2))
    
    elif args.command == "version":
        if not args.version_tag:
            print("Error: --version-tag required for versioning")
            return 1
        deployer = DeploymentManager()
        deployer.version_models(args.version_tag)
        print(f"Models versioned as {args.version_tag}")


if __name__ == "__main__":
    exit(main())
