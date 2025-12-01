import os
import sys
import subprocess
import joblib
import numpy as np


def repo_root():
    # tests/ is directly under the repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_create_and_load_models():
    """Run the dummy model generation script and ensure models can be loaded and predict."""
    root = repo_root()
    script = os.path.join(root, 'scripts', 'create_dummy_models_and_features.py')
    assert os.path.exists(script), f"Expected script at {script}"

    # Run the script to (re)create models and feature CSV
    subprocess.check_call([sys.executable, script], cwd=root)

    model_short = os.path.join(root, 'models', 'model_short.pkl')
    model_long = os.path.join(root, 'models', 'model_long.pkl')
    assert os.path.exists(model_short), "model_short.pkl not found"
    assert os.path.exists(model_long), "model_long.pkl not found"

    # Load and run a single predict to ensure models work
    m_short = joblib.load(model_short)
    m_long = joblib.load(model_long)

    X = np.zeros((1, 10))
    for m in (m_short, m_long):
        pred = m.predict(X)
        assert np.isfinite(pred).all(), "Prediction contains non-finite values"
