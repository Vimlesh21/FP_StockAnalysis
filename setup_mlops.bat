@echo off
REM TCS Stock Forecast - MLOps Quick Start Script (Windows)
REM Sets up the environment and starts all services

echo.
echo ==========================================
echo TCS Stock Forecast - MLOps Setup
echo ==========================================
echo.

REM 1. Check Python version
echo [1] Checking Python version...
python --version

REM 2. Create virtual environment
echo.
echo [2] Setting up Python virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM 3. Install dependencies
echo.
echo [3] Installing Python dependencies...
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

REM 4. Create necessary directories
echo.
echo [4] Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "predictions" mkdir predictions
if not exist "models" mkdir models
if not exist "reports" mkdir reports
if not exist "mlops\runs" mkdir mlops\runs
if not exist "mlops\health" mkdir mlops\health
if not exist "mlops\metrics" mkdir mlops\metrics
if not exist "mlops\model_archive" mkdir mlops\model_archive

REM 5. Check environment variables
echo.
echo [5] Checking environment variables...
if exist ".env" (
    echo Loading .env file...
    for /f "delims== tokens=1,*" %%A in (.env) do set "%%A=%%B"
) else (
    echo Warning: .env file not found
    echo Creating template .env file...
    (
        echo # Database Configuration
        echo DATABASE_URL=postgresql://forecast_user:forecast_pass@localhost:5432/forecast_db
        echo.
        echo # Application Settings
        echo TICKER=TCS.NS
        echo LOG_LEVEL=INFO
        echo PYTHONPATH=.
    ) > .env.template
    echo Template created at .env.template
)

REM 6. Display available commands
echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Available Commands:
echo.
echo [1] Run MLOps Pipeline:
echo     python -m mlops.pipeline --ticker TCS.NS
echo.
echo [2] Start Streamlit Dashboard:
echo     streamlit run streamlit_app.py
echo.
echo [3] Start FastAPI Server:
echo     uvicorn src.api.predict_simple:app --reload
echo.
echo [4] Run Health Check:
echo     python -m mlops.monitoring health
echo.
echo [5] Use Docker (if installed):
echo     docker-compose up -d
echo     Dashboard: http://localhost:8501
echo     API: http://localhost:8000/docs
echo.
echo For detailed documentation, see: MLOPS_GUIDE.md
echo.
