"""Create simple dummy scikit-learn models and a features CSV for local development.
This script is safe to run repeatedly — it will overwrite models and the features CSV.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'return_lag1', 'return_lag2', 'return_lag3', 'return_lag4', 'return_lag5',
    'sma_7', 'sma_21', 'mom_7', 'vol_14', 'volume'
]

# Create dummy training data
rng = np.random.default_rng(42)
X = rng.normal(size=(200, len(FEATURE_COLS)))
# small returns between -0.05 and 0.05
y_short = rng.uniform(-0.05, 0.05, size=(200,))
y_long = rng.uniform(-0.2, 0.2, size=(200,))

# Train simple linear models
short_model = LinearRegression()
short_model.fit(X, y_short)

long_model = LinearRegression()
long_model.fit(X, y_long)

# Save models
joblib.dump(short_model, MODELS_DIR / "model_short.pkl")
joblib.dump(long_model, MODELS_DIR / "model_long.pkl")

print(f"Saved dummy short model to {MODELS_DIR / 'model_short.pkl'}")
print(f"Saved dummy long model to {MODELS_DIR / 'model_long.pkl'}")

# Create a latest-features CSV with one row
values = {
    'date': pd.Timestamp.now().date(),
    'return_lag1': 0.01,
    'return_lag2': -0.005,
    'return_lag3': 0.002,
    'return_lag4': 0.001,
    'return_lag5': -0.003,
    'sma_7': 100.5,
    'sma_21': 99.8,
    'mom_7': 0.012,
    'vol_14': 0.11,
    'volume': 950000
}

df = pd.DataFrame([values])
df.to_csv(DATA_DIR / 'features_latest.csv', index=False)
print(f"Saved features CSV to {DATA_DIR / 'features_latest.csv'}")
