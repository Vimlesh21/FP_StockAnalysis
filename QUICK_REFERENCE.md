# Quick Reference Card - MLOps & Streamlit Deployment

## 🚀 Quick Start

### Docker (30 seconds)
```bash
docker-compose up -d
# Dashboard: http://localhost:8501
# API: http://localhost:8000/docs
```

### Local (Linux/macOS)
```bash
bash setup_mlops.sh
streamlit run streamlit_app.py
```

### Local (Windows)
```cmd
setup_mlops.bat
streamlit run streamlit_app.py
```

---

## 📊 Main URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit Dashboard** | http://localhost:8501 | Monitoring & predictions |
| **FastAPI Docs** | http://localhost:8000/docs | API documentation |
| **Database** | localhost:5432 | PostgreSQL |

---

## 🔧 Common Commands

### MLOps Pipeline
```bash
# Run full pipeline
python -m mlops.pipeline --ticker TCS.NS

# Skip training
python -m mlops.pipeline --ticker TCS.NS --skip-training

# With custom config
python -m mlops.pipeline --config config.json
```

### Monitoring
```bash
# Health check
python -m mlops.monitoring health

# Metrics report
python -m mlops.monitoring metrics

# Version models
python -m mlops.monitoring version --version-tag v1.0

# Rollback
python -m mlops.monitoring version --rollback v0.9
```

### FastAPI
```bash
# Start API server
uvicorn src.api.predict_simple:app --reload

# Test endpoint
curl -X POST http://localhost:8000/predict/short
```

### Docker
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f dashboard

# Stop services
docker-compose down

# Rebuild images
docker-compose build
```

---

## 📈 Dashboard Pages

1. **Home** - Overview & status
2. **Predictions** - Short/long-term forecasts
3. **Analytics** - Model metrics & history
4. **Pipeline Control** - Run workflows
5. **Monitoring** - Health & logs

---

## 📁 Key Directories

| Path | Contents |
|------|----------|
| `mlops/` | Pipeline, monitoring, runs |
| `src/api/` | FastAPI application |
| `src/data/` | Data layer & cleaning |
| `src/features/` | Feature engineering |
| `predictions/` | Prediction outputs |
| `models/` | Trained models |
| `reports/` | Evaluation metrics |
| `logs/` | Application logs |

---

## 🔄 Data Pipeline Flow

```
Raw Data
   ↓
[Data Cleaning] → raw_*_clean tables
   ↓
[Feature Engineering] → features_daily table
   ↓
[Validation] → Quality checks
   ↓
[Training] → Models (short & long)
   ↓
[Prediction] → Results in API/Dashboard
```

---

## 📋 Environment Variables

```env
DATABASE_URL=postgresql://user:pass@host:5432/db
TICKER=TCS.NS
LOG_LEVEL=INFO
PYTHONPATH=.
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Dashboard won't start | `pip install streamlit==1.28.1` |
| API connection error | Check `DATABASE_URL` env var |
| Docker containers fail | `docker-compose logs postgres` |
| Pipeline fails | `python -m mlops.monitoring health` |
| Database not accessible | `docker-compose ps` |

---

## 🎯 Pipeline Stages

1. **Data Cleaning** - Load & validate raw data
2. **Features** - Compute technical indicators
3. **Validation** - Quality & completeness checks
4. **Training** - (Optional) Train models
5. **Prediction** - Generate forecasts

---

## 🔍 Health Check Status

```bash
python -m mlops.monitoring health
```

Checks:
- ✓ Database connectivity
- ✓ Model availability
- ✓ Feature data freshness
- ✓ Data quality score
- ⚠️ Alert summary

---

## 📊 Model Versioning

```bash
# Version current models
python -m mlops.monitoring version --version-tag v1.0-2025-01-30

# List available versions
ls mlops/model_archive/

# Rollback to version
python -m mlops.monitoring version --rollback v0.9-2025-01-14
```

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src

# Specific test
pytest tests/test_pipeline.py -v
```

---

## 📝 Logs Location

```
logs/tcs-stock.log          # Main application log
mlops/runs/{ID}/metrics.json    # Pipeline metrics
mlops/runs/{ID}/results.json    # Pipeline results
mlops/health/health_*.json  # Health reports
```

---

## 🐳 Docker Services

```
postgres (5432)     - Database
api (8000)         - FastAPI server
dashboard (8501)   - Streamlit UI
pipeline           - MLOps orchestrator (optional)
```

---

## 🚀 Deployment Checklist

- [ ] Environment variables set (.env file)
- [ ] Database initialized and accessible
- [ ] Docker images built
- [ ] Services started with docker-compose
- [ ] Dashboard accessible at port 8501
- [ ] API responding at port 8000
- [ ] Health checks passing
- [ ] Pipeline run completed

---

## 📚 Documentation Files

- **`.github/copilot-instructions.md`** - Architecture overview
- **`MLOPS_GUIDE.md`** - Detailed technical guide
- **`MLOPS_README.md`** - Quick start & usage
- **`IMPLEMENTATION_SUMMARY.md`** - What's been created
- **`QUICK_REFERENCE.md`** - This file

---

## 🔗 API Endpoints

```
POST /predict/short      - Next-day prediction
POST /predict/long       - 63-day prediction
GET  /predict/short      - Browser-friendly short
GET  /predict/long       - Browser-friendly long
GET  /health             - System health check
```

---

## 💡 Pro Tips

1. **Fast Development**: Use `--skip-training` to test data pipeline only
2. **Debugging**: Check `logs/tcs-stock.log` for detailed errors
3. **Performance**: Run pipeline during off-hours with cron
4. **Monitoring**: Check Dashboard → Monitoring page daily
5. **Backups**: Version models after successful training
6. **Rollback**: Keep at least 2 versions for safety

---

## 🎓 Learning Path

1. Start with Dashboard (Home page)
2. Review API docs (localhost:8000/docs)
3. Run pipeline manually once
4. Check logs and metrics
5. Explore monitoring page
6. Read MLOPS_GUIDE.md for details

---

**Last Updated**: January 30, 2025  
**Version**: 1.0.0
