"""
FastAPI Prediction Endpoints
Provides short-term (1-day) and long-term (~63-day) stock price predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Feature columns that must match training data
FEATURE_COLS = [
    'return_lag1', 'return_lag2', 'return_lag3', 'return_lag4', 'return_lag5',
    'sma_7', 'sma_21', 'mom_7', 'vol_14', 'volume'
]

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="TCS Stock Forecast API",
    description="Prediction API for TCS.NS stock prices",
    version="1.0.0"
)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models() -> Dict[str, Any]:
    """Load trained models from disk."""
    models = {}
    try:
        short_path = MODELS_DIR / "model_short.pkl"
        long_path = MODELS_DIR / "model_long.pkl"
        
        if short_path.exists():
            models['short'] = joblib.load(short_path)
            logger.info(f"Loaded short-term model from {short_path}")
        else:
            logger.warning(f"Short-term model not found at {short_path}")
            
        if long_path.exists():
            models['long'] = joblib.load(long_path)
            logger.info(f"Loaded long-term model from {long_path}")
        else:
            logger.warning(f"Long-term model not found at {long_path}")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
    
    return models

# Load models on startup
_MODELS = load_models()

# ============================================================================
# FEATURE DATA FUNCTIONS
# ============================================================================

def get_last_feature_row() -> Optional[pd.DataFrame]:
    """
    Retrieve the last row from the features table.
    Attempts to load from CSV or database.
    """
    try:
        # Try loading from CSV
        features_csv = PROJECT_ROOT / "data" / "features_latest.csv"
        if features_csv.exists():
            df = pd.read_csv(features_csv)
            if len(df) > 0:
                return df.iloc[-1:]
    except Exception as e:
        logger.warning(f"Could not load features from CSV: {e}")
    
    return None

def create_dummy_features() -> pd.Series:
    """Create dummy feature values for demonstration."""
    return pd.Series({
        'date': datetime.now().date(),
        'return_lag1': 0.01,
        'return_lag2': -0.005,
        'return_lag3': 0.008,
        'return_lag4': 0.002,
        'return_lag5': -0.003,
        'sma_7': 100.5,
        'sma_21': 99.8,
        'mom_7': 0.015,
        'vol_14': 0.12,
        'volume': 1000000
    })

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_short(features: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Predict 1-day ahead stock return.
    
    Args:
        features: Feature vector. If None, uses latest from database.
    
    Returns:
        Dict with prediction, confidence, and metadata.
    """
    try:
        if features is None:
            feature_row = get_last_feature_row()
            if feature_row is not None:
                features = feature_row.iloc[0]
            else:
                # Use dummy features for demonstration
                features = create_dummy_features()
        
        # Check if model is available
        if 'short' not in _MODELS or _MODELS['short'] is None:
            return {
                "horizon": "1-day",
                "feature_date": str(features.get('date', datetime.now().date())),
                "predict_date": str((datetime.now() + timedelta(days=1)).date()),
                "pred_return": 0.0,
                "est_future_close": 0.0,
                "pct_return": 0.0,
                "note": "Model not available - returning dummy prediction",
                "warning": "Model training required"
            }
        
        # Extract features in correct order
        X = features[FEATURE_COLS].values.reshape(1, -1)
        
        # Make prediction
        pred_return = _MODELS['short'].predict(X)[0]
        
        # Estimate future close (assuming current close = 100)
        current_close = 100.0
        est_future_close = current_close * (1 + pred_return)
        pct_return = pred_return * 100
        
        feature_date = features.get('date', datetime.now().date())
        predict_date = (datetime.now() + timedelta(days=1)).date()
        
        return {
            "horizon": "1-day",
            "feature_date": str(feature_date),
            "predict_date": str(predict_date),
            "pred_return": float(pred_return),
            "est_future_close": float(est_future_close),
            "pct_return": float(pct_return),
            "note": "Prediction based on latest available features"
        }
        
    except Exception as e:
        logger.error(f"Error in predict_short: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def predict_long(features: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Predict ~63-day ahead stock return.
    
    Args:
        features: Feature vector. If None, uses latest from database.
    
    Returns:
        Dict with prediction, confidence, and metadata.
    """
    try:
        if features is None:
            feature_row = get_last_feature_row()
            if feature_row is not None:
                features = feature_row.iloc[0]
            else:
                # Use dummy features for demonstration
                features = create_dummy_features()
        
        # Check if model is available
        if 'long' not in _MODELS or _MODELS['long'] is None:
            return {
                "horizon": "63-day",
                "feature_date": str(features.get('date', datetime.now().date())),
                "predict_date": str((datetime.now() + timedelta(days=63)).date()),
                "pred_return": 0.0,
                "est_future_close": 0.0,
                "pct_return": 0.0,
                "note": "Model not available - returning dummy prediction",
                "warning": "Model training required"
            }
        
        # Extract features in correct order
        X = features[FEATURE_COLS].values.reshape(1, -1)
        
        # Make prediction
        pred_return = _MODELS['long'].predict(X)[0]
        
        # Estimate future close
        current_close = 100.0
        est_future_close = current_close * (1 + pred_return)
        pct_return = pred_return * 100
        
        feature_date = features.get('date', datetime.now().date())
        predict_date = (datetime.now() + timedelta(days=63)).date()
        
        return {
            "horizon": "63-day",
            "feature_date": str(feature_date),
            "predict_date": str(predict_date),
            "pred_return": float(pred_return),
            "est_future_close": float(est_future_close),
            "pct_return": float(pct_return),
            "note": "Prediction based on latest available features"
        }
        
    except Exception as e:
        logger.error(f"Error in predict_long: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "short": 'short' in _MODELS and _MODELS['short'] is not None,
            "long": 'long' in _MODELS and _MODELS['long'] is not None
        }
    }

@app.post("/predict/short")
def api_predict_short():
    """API endpoint for short-term prediction."""
    return predict_short()

@app.post("/predict/long")
def api_predict_long():
    """API endpoint for long-term prediction."""
    return predict_long()

@app.get("/predict/short")
def api_predict_short_get():
    """Browser-friendly GET endpoint for short-term prediction."""
    return predict_short()

@app.get("/predict/long")
def api_predict_long_get():
    """Browser-friendly GET endpoint for long-term prediction."""
    return predict_long()

@app.get("/")
def root():
    """Root endpoint with API documentation."""
    return {
        "name": "TCS Stock Forecast API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "short_prediction": "/predict/short",
            "long_prediction": "/predict/long"
        }
    }

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("TCS Stock Forecast API starting...")
    logger.info(f"Models loaded: {list(_MODELS.keys())}")
    logger.info(f"Feature columns: {FEATURE_COLS}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("TCS Stock Forecast API shutting down...")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
