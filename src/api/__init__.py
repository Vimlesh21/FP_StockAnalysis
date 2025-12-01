"""TCS Stock Forecast API Module"""

from .predict_simple import (
    app,
    predict_short,
    predict_long,
    get_last_feature_row,
    FEATURE_COLS
)

__all__ = [
    'app',
    'predict_short',
    'predict_long',
    'get_last_feature_row',
    'FEATURE_COLS'
]
