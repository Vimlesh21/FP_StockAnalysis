# TCS Stock Forecast - MLOps & Streamlit Deployment

Complete production-ready MLOps pipeline with interactive Streamlit dashboard for stock price forecasting.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# Access services
# Dashboard: http://localhost:8501
# API Docs: http://localhost:8000/docs
# Database: localhost:5432

# View logs
docker-compose logs -f dashboard
```

### Option 2: Local Setup (Linux/macOS)

```bash
# Run setup script
bash setup_mlops.sh

# Start services (in separate terminals)
streamlit run streamlit_app.py          # Dashboard
uvicorn src.api.predict_simple:app --reload  # API
python -m mlops.pipeline --ticker TCS.NS     # Pipeline
```

### Option 3: Local Setup (Windows)

```cmd
# Run setup script
setup_mlops.bat

# Start services (in separate terminals)
streamlit run streamlit_app.py
uvicorn src.api.predict_simple:app --reload
python -m mlops.pipeline --ticker TCS.NS
```

## 📚 Key Components

### 1. MLOps Pipeline (`mlops/pipeline.py`)
Automated workflow orchestrating data, features, validation, and training:

```bash
python -m mlops.pipeline --ticker TCS.NS [--skip-training] [--config config.json]
```

**Stages:**
- Data Cleaning: Raw data → cleaned tables
- Feature Engineering: Technical indicators computation
- Data Validation: Quality checks and completeness
- Training: Model development (optional)

**Output:** Metrics, run logs, and results saved to `mlops/runs/{TIMESTAMP}/`

### 2. Streamlit Dashboard (`streamlit_app.py`)
Interactive web UI with 5 main pages:

| Page | Features |
|------|----------|
| **Home** | System status, latest predictions, quick links |
| **Predictions** | Short/long-term forecasts, comparison view |
| **Analytics** | Model performance, prediction history, features |
| **Pipeline Control** | Run pipeline, view recent executions |
| **Monitoring** | Health checks, logs, system diagnostics |

```bash
streamlit run streamlit_app.py
# Access at http://localhost:8501
```

### 3. Monitoring System (`mlops/monitoring.py`)
Health checks, metrics collection, and model versioning:

```python
from mlops.monitoring import ModelMonitor, MetricsCollector, DeploymentManager

# Health check
monitor = ModelMonitor()
health = monitor.run_health_check()

# Metrics
collector = MetricsCollector()
collector.save_metric("model_r2", 0.45)

# Versioning
deployer = DeploymentManager()
deployer.version_models("v1.0")
```

CLI:
```bash
python -m mlops.monitoring health                        # Health check
python -m mlops.monitoring metrics                       # Metrics report
python -m mlops.monitoring version --version-tag v1.0    # Version models
```

### 4. FastAPI Backend (`src/api/predict_simple.py`)
RESTful API for predictions and health checks:

```bash
uvicorn src.api.predict_simple:app --reload
# Swagger UI: http://localhost:8000/docs
```

**Endpoints:**
- `POST /predict/short` - Next-day prediction
- `POST /predict/long` - 63-day prediction
- `GET /predict/short`, `GET /predict/long` - Browser-friendly
- `GET /health` - System health

## 📊 Architecture & Data Flow

```
Data Sources
    ↓
[MLOps Pipeline] → [Data Cleaning] → [Feature Engineering] → [Validation]
    ↓
[PostgreSQL Database]
    ↓
├─→ [Streamlit Dashboard] (Port 8501)
│   ├─ Live predictions
│   ├─ Analytics & charts
│   ├─ Pipeline control
│   └─ System monitoring
│
├─→ [FastAPI Backend] (Port 8000)
│   ├─ /predict/short
│   ├─ /predict/long
│   └─ /health
│
└─→ [Monitoring System]
    ├─ Health checks
    ├─ Model versioning
    └─ Metrics collection
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/forecast_db

# Application
TICKER=TCS.NS
LOG_LEVEL=INFO
PYTHONPATH=.
```

### Pipeline Configuration

Create `config.json`:
```json
{
  "data": {
    "ticker": "TCS.NS",
    "min_rows_required": 100
  },
  "features": {
    "lag_periods": [1, 2, 3, 4, 5],
    "sma_periods": [7, 21],
    "momentum_period": 7,
    "volatility_period": 14
  },
  "validation": {
    "min_mae_improvement": 0.01,
    "min_r2_improvement": 0.05
  }
}
```

## 📁 Directory Structure

```
tcsns-forecast/
├── mlops/
│   ├── pipeline.py              # Pipeline orchestrator
│   ├── monitoring.py            # Health & monitoring
│   ├── runs/                    # Pipeline execution logs
│   ├── health/                  # Health reports
│   ├── metrics/                 # Time-series metrics
│   └── model_archive/           # Versioned models
├── src/
│   ├── api/predict_simple.py    # FastAPI endpoints
│   ├── data/                    # Data layer (cleaning, DB)
│   ├── features/                # Feature engineering
│   └── config/                  # Configuration
├── streamlit_app.py             # Streamlit dashboard
├── docker-compose.yml           # Container orchestration
├── Dockerfile                   # Streamlit image
├── Dockerfile.api               # FastAPI image
├── requirements.txt             # Python dependencies
├── MLOPS_GUIDE.md               # Detailed documentation
├── setup_mlops.sh               # Linux/macOS setup
└── setup_mlops.bat              # Windows setup
```

## 🐳 Docker Deployment

### Services

**docker-compose.yml** includes:

1. **PostgreSQL** (port 5432)
   - Persistent data storage
   - Health checks enabled
   
2. **FastAPI** (port 8000)
   - REST predictions endpoint
   - Auto-reload on code changes
   
3. **Streamlit Dashboard** (port 8501)
   - Interactive web UI
   - Real-time monitoring
   
4. **MLOps Pipeline** (optional)
   - Run with `--profile pipeline`
   - Executes scheduled workflows

### Common Commands

```bash
# Start all services
docker-compose up -d

# Start with pipeline
docker-compose --profile pipeline up -d

# View specific service logs
docker-compose logs -f dashboard
docker-compose logs -f api
docker-compose logs -f postgres

# Stop all services
docker-compose down

# Remove data (clean reset)
docker-compose down -v

# Rebuild images
docker-compose build
```

## 📈 Usage Examples

### Running Full Pipeline

```bash
python -m mlops.pipeline --ticker TCS.NS
```

Output:
```
============================================================
MLOPS PIPELINE RUN: 20250130_143022
============================================================

[data_cleaning] Status: success, Rows: 2471, Duration: 2.45s
[feature_engineering] Status: success, Rows: 2409, Duration: 1.23s
[validation] Status: success, Rows: 2409, Duration: 0.89s

============================================================
PIPELINE COMPLETED SUCCESSFULLY
Total duration: 4.57s
============================================================
```

### Getting Predictions

```bash
# Using curl
curl -X POST http://localhost:8000/predict/short
curl -X POST http://localhost:8000/predict/long

# Using Python
import requests
response = requests.post("http://localhost:8000/predict/short")
print(response.json())
```

### System Health Check

```bash
python -m mlops.monitoring health
```

Output:
```json
{
  "timestamp": "2025-01-30T14:30:22.123456",
  "database_connected": true,
  "models_available": true,
  "features_stale": false,
  "data_quality_score": 92.5,
  "alerts": []
}
```

### Model Versioning

```bash
# Version current models
python -m mlops.monitoring version --version-tag v1.0-2025-01-30

# List versions
ls mlops/model_archive/

# Rollback
python -m mlops.monitoring version --rollback v0.9
```

## 🎯 Typical Workflows

### Daily Data Refresh

1. Data arrives (external source)
2. Run pipeline: `python -m mlops.pipeline --ticker TCS.NS`
3. Check dashboard for updates
4. Review health metrics

### Model Retraining

1. Run full pipeline: `python -m mlops.pipeline --ticker TCS.NS`
2. Check evaluation metrics in Analytics page
3. If improved, version models: `python -m mlops.monitoring version`
4. If not, rollback: `python -m mlops.monitoring version --rollback`

### Monitoring & Alerts

1. Dashboard → Monitoring → System Health
2. Check for critical alerts
3. Review logs if issues found
4. Run health check: `python -m mlops.monitoring health`

## 🚨 Troubleshooting

### Dashboard not starting
```bash
# Check logs
streamlit run streamlit_app.py --logger.level=debug

# Verify dependencies
pip install -r requirements.txt --upgrade
```

### API errors
```bash
# Check database connection
python -c "from src.data.db_utils import get_engine; get_engine()"

# View logs
tail -f logs/tcs-stock.log
```

### Docker issues
```bash
# Check service status
docker-compose ps

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Pipeline fails
1. Check logs: `mlops/runs/{TIMESTAMP}/metrics.json`
2. Run health check: `python -m mlops.monitoring health`
3. Verify database: `docker-compose exec postgres pg_isready`

## 📊 Performance Tuning

### Database Optimization
```sql
-- Create indexes for faster queries
CREATE INDEX idx_features_date ON features_daily(date);
CREATE INDEX idx_dataset_short_date ON dataset_short(date);
CREATE INDEX idx_dataset_long_date ON dataset_long(date);
```

### Streamlit Caching
- `@st.cache_data` for expensive computations
- `@st.cache_resource` for database connections
- Clear cache: `streamlit cache clear`

### API Performance
- Use connection pooling (SQLAlchemy default)
- Batch predictions when possible
- Monitor response times in logs

## 🔐 Security Best Practices

1. **Environment Variables**: Never commit `.env` files
2. **Database**: Use strong passwords, restrict network access
3. **API**: Add authentication (JWT, API keys)
4. **Logs**: Avoid logging sensitive data
5. **Docker**: Run as non-root user in production

## 📚 Additional Resources

- **[MLOPS_GUIDE.md](MLOPS_GUIDE.md)** - Detailed MLOps documentation
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Architecture overview
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **PostgreSQL Docs**: https://www.postgresql.org/docs/

## 🤝 Contributing

To extend the system:

1. **Add new features**: Update `src/features/features_daily.py`
2. **Add new models**: Extend `src/models/` directory
3. **Extend pipeline**: Modify `mlops/pipeline.py`
4. **Add dashboard pages**: Update `streamlit_app.py`

Remember to:
- Update `FEATURE_COLS` in `src/api/predict_simple.py` if adding features
- Test pipeline locally before Docker
- Run health checks after changes
- Version models before deploying

## 📝 License

This project is part of ISB AMPBA Foundation Project.

## 📧 Support

For issues or questions:
1. Check logs: `logs/tcs-stock.log`
2. Review health reports: `mlops/health/`
3. Inspect pipeline runs: `mlops/runs/`
4. Check monitoring metrics: `mlops/metrics/`

---

**Last Updated**: January 30, 2025  
**Version**: 1.0.0
