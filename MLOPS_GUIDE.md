# MLOps Pipeline & Streamlit Deployment Guide

## Overview

This document describes the complete MLOps infrastructure for the TCS Stock Forecast system, including:
- **MLOps Pipeline**: Automated data, feature, training, and evaluation workflow
- **Streamlit Dashboard**: Interactive web UI for monitoring and predictions
- **Monitoring System**: Health checks, alerts, and metrics tracking
- **Docker Deployment**: Containerized services for easy deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Sources (External)                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
      ┌──────────▼──────────┐
      │  MLOps Pipeline     │ (mlops/pipeline.py)
      │  ┌────────────────┐ │
      │  │ Data Cleaning  │ │
      │  ├────────────────┤ │
      │  │ Features       │ │
      │  ├────────────────┤ │
      │  │ Validation     │ │
      │  ├────────────────┤ │
      │  │ Training*      │ │
      │  └────────────────┘ │
      └──────────┬──────────┘
                 │
      ┌──────────▼──────────────────┐
      │   PostgreSQL Database       │
      │  - raw_daily_prices         │
      │  - raw_daily_prices_clean   │
      │  - features_daily           │
      │  - dataset_short            │
      │  - dataset_long             │
      └──────────┬──────────────────┘
                 │
      ┌──────────┴─────────────────────────┐
      │                                    │
  ┌───▼────────────┐        ┌────────────▼──┐
  │ FastAPI        │        │  Streamlit    │
  │ (Port 8000)    │        │  (Port 8501)  │
  │ - /predict/*   │        │ Dashboard UI  │
  │ - /health      │        │ Monitoring    │
  └────────────────┘        └───────────────┘
      │
      └──────────┬──────────────────┐
                 │                  │
         ┌───────▼────────┐    ┌───▼──────────┐
         │  Monitoring    │    │ Model        │
         │  System        │    │ Versioning   │
         │ (Health Checks)│    │ & Rollback   │
         └────────────────┘    └──────────────┘
```

## MLOps Pipeline

### 1. Pipeline Orchestrator (`mlops/pipeline.py`)

The `MLOpsPipeline` class orchestrates the complete ML workflow:

```python
from mlops.pipeline import MLOpsPipeline

# Initialize pipeline
pipeline = MLOpsPipeline(ticker="TCS.NS")

# Run full pipeline (data → features → validation → training)
success, results = pipeline.run_full_pipeline(skip_training=False)

# Check results
if success:
    print(f"Pipeline completed: {results['total_duration_seconds']:.2f}s")
else:
    print(f"Pipeline failed: {results['error']}")
```

### 2. Pipeline Stages

#### Stage 1: Data Cleaning
- Loads raw daily and hourly prices from database
- Removes duplicates, fills missing values
- Validates data integrity
- Output: `raw_daily_prices_clean`, `raw_hourly_prices_clean`

#### Stage 2: Feature Engineering
- Computes technical indicators from cleaned daily data
- Generates lag returns, moving averages, momentum, volatility
- Feature columns: `return_lag1-5`, `sma_7`, `sma_21`, `mom_7`, `vol_14`, `volume`
- Output: `features_daily` table

#### Stage 3: Data Validation
- Verifies feature data quality and completeness
- Checks minimum row requirements
- Validates feature column names and types
- Triggers alerts if quality is below threshold

#### Stage 4: Model Training (Optional)
- Builds datasets for short (1-day) and long (63-day) horizons
- Trains scikit-learn models
- Evaluates performance on test set
- Saves models and metrics

### 3. Running the Pipeline

**CLI Usage:**
```bash
# Full pipeline with training
python -m mlops.pipeline --ticker TCS.NS

# Skip training stage (data refresh only)
python -m mlops.pipeline --ticker TCS.NS --skip-training

# Use custom config
python -m mlops.pipeline --ticker TCS.NS --config config.json
```

**Configuration Example (`config.json`):**
```json
{
  "data": {
    "ticker": "TCS.NS",
    "min_rows_required": 100
  },
  "features": {
    "lag_periods": [1, 2, 3, 4, 5],
    "sma_periods": [7, 21]
  },
  "validation": {
    "min_mae_improvement": 0.01
  }
}
```

### 4. Pipeline Outputs

Each pipeline run creates:
- `mlops/runs/{TIMESTAMP}/results.json` - Pipeline results
- `mlops/runs/{TIMESTAMP}/metrics.json` - Stage-by-stage metrics

## Streamlit Dashboard

### 1. Launch Dashboard

```bash
# Local development
streamlit run streamlit_app.py

# Docker container
docker-compose up dashboard
```

Access at: `http://localhost:8501`

### 2. Dashboard Pages

#### 🏠 Home
- Overview metrics (latest short/long predictions)
- System status (models, database, evaluation)
- Quick links to API and reports

#### 🔮 Predictions
- **Short-term tab**: Next-day return prediction with metrics
- **Long-term tab**: 63-day return forecast
- **Comparison tab**: Side-by-side short vs. long predictions

#### 📊 Analytics
- **Model Performance**: MAE, RMSE, R² score comparison
- **Predictions History**: Time-series plot of actual vs. predicted
- **Feature Analysis**: Feature column descriptions

#### ⚙️ Pipeline Control
- Run full MLOps pipeline from UI
- Monitor execution progress
- View recent pipeline runs and results

#### 📋 Monitoring
- **Pipeline Runs**: History of all executions
- **System Health**: Database, models, evaluation status
- **Logs**: Real-time application logs with download

### 3. Key Features

- **Real-time Predictions**: Fetches latest predictions via API
- **Interactive Charts**: Plotly-based visualizations
- **Performance Metrics**: Comprehensive model evaluation
- **Pipeline Execution**: Trigger and monitor MLOps workflows
- **Health Monitoring**: System status and alerts
- **Log Viewer**: Tail and download application logs

## Monitoring System

### 1. Health Checks (`mlops/monitoring.py`)

```python
from mlops.monitoring import ModelMonitor

monitor = ModelMonitor()

# Run comprehensive health check
health_check = monitor.run_health_check()
# Returns: HealthCheck with alerts, data quality, model status

# Save report
monitor.save_health_report(health_check)

# Individual checks
db_ok, msg = monitor.check_database_connection()
stale, msg = monitor.check_model_staleness(threshold_days=7)
quality, details = monitor.check_feature_data_quality()
perf_score, details = monitor.check_model_performance_drift()
```

### 2. Metrics Collection

```python
from mlops.monitoring import MetricsCollector

collector = MetricsCollector()

# Save metric point
collector.save_metric("model_r2_score", 0.45, tags={"horizon": "short"})
collector.save_metric("data_quality", 92.5, tags={"table": "features"})

# Get historical metrics
history = collector.get_metric_history("model_r2_score", days=7)

# Generate report
report = collector.generate_metrics_report()
```

### 3. Model Versioning & Rollback

```python
from mlops.monitoring import DeploymentManager

deployer = DeploymentManager()

# Version current models
deployer.version_models("v1.0-2025-01-15")

# List all versions
versions = deployer.list_versions()
# ['v1.0-2025-01-15', 'v0.9-2025-01-14', ...]

# Rollback to previous version
deployer.rollback_to_version("v0.9-2025-01-14")
```

### 4. Alert Severity Levels

- **INFO**: Informational messages
- **WARNING**: Needs attention (stale data, performance degradation)
- **CRITICAL**: Requires immediate action (database down, model missing)

## Docker Deployment

### 1. Services

**docker-compose.yml** defines:
- **postgres**: PostgreSQL database (port 5432)
- **api**: FastAPI service (port 8000)
- **dashboard**: Streamlit (port 8501)
- **pipeline**: MLOps pipeline (scheduled)

### 2. Quick Start

```bash
# Start all services (except pipeline)
docker-compose up -d

# Start with pipeline
docker-compose --profile pipeline up -d

# View logs
docker-compose logs -f dashboard

# Stop services
docker-compose down
```

### 3. Environment Variables

Create `.env` file:
```
DATABASE_URL=postgresql://forecast_user:forecast_pass@postgres:5432/forecast_db
TICKER=TCS.NS
LOG_LEVEL=INFO
```

### 4. Building Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build dashboard

# Build and push to registry
docker build -t myregistry/tcs-forecast:latest .
docker push myregistry/tcs-forecast:latest
```

## Typical Workflows

### 1. Daily Data Refresh

```bash
# Run pipeline (automatically refreshes data and features)
python -m mlops.pipeline --ticker TCS.NS

# Check dashboard
# http://localhost:8501 → Pipeline Control → View Recent Runs
```

### 2. Model Retraining

```bash
# Full pipeline includes training
python -m mlops.pipeline --ticker TCS.NS

# Check metrics
# http://localhost:8501 → Analytics → Model Performance
```

### 3. Health Check & Monitoring

```bash
# CLI health check
python -m mlops.monitoring health

# Streamlit monitoring
# http://localhost:8501 → Monitoring → System Health

# View metrics report
python -m mlops.monitoring metrics
```

### 4. Model Rollback

```bash
# List available versions
python -m mlops.monitoring version --list

# Rollback to previous version
python -m mlops.monitoring rollback --version-tag v0.9-2025-01-14
```

### 5. API Predictions

```bash
# Short-term prediction
curl -X POST http://localhost:8000/predict/short

# Long-term prediction
curl -X POST http://localhost:8000/predict/long

# Browser access
# http://localhost:8000/docs (Swagger UI)
```

## Key Files & Locations

```
tcsns-forecast/
├── mlops/
│   ├── pipeline.py           # MLOps orchestrator
│   ├── monitoring.py         # Health checks & metrics
│   ├── runs/                 # Pipeline execution logs
│   ├── health/               # Health check reports
│   ├── metrics/              # Time-series metrics
│   └── model_archive/        # Versioned models
├── streamlit_app.py          # Streamlit dashboard
├── docker-compose.yml        # Container orchestration
├── Dockerfile                # Streamlit image
├── Dockerfile.api            # FastAPI image
├── requirements.txt          # Python dependencies
└── src/
    ├── api/predict_simple.py # FastAPI endpoints
    ├── data/                 # Data layer
    ├── features/             # Feature engineering
    └── config/               # Configuration
```

## Troubleshooting

### Pipeline Fails at Data Stage
1. Check database connection: `python -m mlops.monitoring health`
2. Verify raw data exists: `SELECT COUNT(*) FROM raw_daily_prices;`
3. Check logs: `tail -f logs/tcs-stock.log`

### Models Not Available in Dashboard
1. Check if models exist: `ls -la models/`
2. Run pipeline: `python -m mlops.pipeline --ticker TCS.NS`
3. Verify paths in `src/api/predict_simple.py`

### High Feature Latency
1. Check feature table: `SELECT MAX(date) FROM features_daily;`
2. If stale, run data refresh: `python -m src.data.cleaning write --ticker TCS.NS`
3. Then run features: `python -m src.features.features_daily write --ticker TCS.NS`

### Docker Services Not Starting
1. Check logs: `docker-compose logs postgres`
2. Verify DB connection: `docker exec tcs_forecast_db pg_isready`
3. Check environment variables in `.env`

## Performance Optimization

### 1. Pipeline Execution
- Skip training for faster data-only refreshes: `--skip-training`
- Increase worker processes for parallel feature computation
- Use connection pooling in database layer

### 2. Streamlit Dashboard
- Cache data with `@st.cache_data`
- Use lazy loading for large tables
- Limit log display to recent lines

### 3. Database Indexing
Create indexes for faster queries:
```sql
CREATE INDEX idx_features_date ON features_daily(date);
CREATE INDEX idx_dataset_short_date ON dataset_short(date);
CREATE INDEX idx_dataset_long_date ON dataset_long(date);
```

## Next Steps

1. **Schedule Pipelines**: Use cron/Airflow for automated daily runs
2. **CI/CD Integration**: Add GitHub Actions for automated testing
3. **Model Registry**: Implement MLflow for centralized model management
4. **Alerting**: Integrate with Slack/PagerDuty for critical alerts
5. **Performance Tuning**: Add A/B testing framework for model comparison
6. **Data Quality**: Implement Great Expectations for data validation

## Support & Debugging

For issues or questions:
1. Check logs: `logs/tcs-stock.log`
2. Review health reports: `mlops/health/`
3. Inspect pipeline runs: `mlops/runs/`
4. Check monitoring metrics: `mlops/metrics/`
