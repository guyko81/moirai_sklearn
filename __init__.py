"""
moirai_sklearn - A scikit-learn-like interface for Moirai time series forecasting.

Uses native quantile outputs to compute mean, median, mode, std, etc.

Usage:
    from moirai_sklearn import MoiraiForecaster
    
    model = MoiraiForecaster()
    predictions = model.predict(my_time_series, horizon=30)
"""

from .forecaster import MoiraiForecaster

__version__ = "0.1.0"
__all__ = ["MoiraiForecaster"]
