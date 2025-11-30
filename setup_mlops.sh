#!/bin/bash

# TCS Stock Forecast - MLOps Quick Start Script
# Sets up the environment and starts all services

set -e

echo "=========================================="
echo "TCS Stock Forecast - MLOps Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check Python version
echo -e "\n${BLUE}1. Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# 2. Check if virtual environment exists, create if not
echo -e "\n${BLUE}2. Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
echo -e "\n${BLUE}3. Installing Python dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Create necessary directories
echo -e "\n${BLUE}4. Creating necessary directories...${NC}"
mkdir -p logs predictions models reports mlops/runs mlops/health mlops/metrics mlops/model_archive

# 5. Check environment variables
echo -e "\n${BLUE}5. Checking environment variables...${NC}"
if [ -f ".env" ]; then
    echo "Loading .env file..."
    export $(cat .env | xargs)
else
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating template .env file..."
    cat > .env.template << 'EOF'
# Database Configuration
DATABASE_URL=postgresql://forecast_user:forecast_pass@localhost:5432/forecast_db

# Application Settings
TICKER=TCS.NS
LOG_LEVEL=INFO
PYTHONPATH=.
EOF
    echo "Template created at .env.template"
fi

# 6. Database connectivity check (optional)
if [ ! -z "$DATABASE_URL" ]; then
    echo -e "\n${BLUE}6. Checking database connectivity...${NC}"
    python -c "
from src.data.db_utils import get_engine
try:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute('SELECT 1')
    print('✓ Database connected successfully')
except Exception as e:
    print(f'✗ Database connection failed: {e}')
" || echo -e "${YELLOW}Database not available - some features will be limited${NC}"
else
    echo -e "${YELLOW}DATABASE_URL not set - skipping database check${NC}"
fi

# 7. Display available commands
echo -e "\n${GREEN}=========================================="
echo "Setup Complete! You can now run:"
echo "==========================================${NC}"

echo -e "\n${BLUE}Option 1: Run MLOps Pipeline${NC}"
echo "  python -m mlops.pipeline --ticker TCS.NS"

echo -e "\n${BLUE}Option 2: Start Streamlit Dashboard${NC}"
echo "  streamlit run streamlit_app.py"

echo -e "\n${BLUE}Option 3: Start FastAPI Server${NC}"
echo "  uvicorn src.api.predict_simple:app --reload"

echo -e "\n${BLUE}Option 4: Run Health Check${NC}"
echo "  python -m mlops.monitoring health"

echo -e "\n${BLUE}Option 5: Use Docker (if installed)${NC}"
echo "  docker-compose up -d"
echo "  # Dashboard: http://localhost:8501"
echo "  # API: http://localhost:8000/docs"

echo -e "\n${GREEN}For detailed documentation, see: MLOPS_GUIDE.md${NC}\n"
