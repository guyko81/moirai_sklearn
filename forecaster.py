"""
Moirai Forecaster - A scikit-learn-like wrapper for Moirai time series models.

Uses the native quantile predictions from Moirai to provide:
- predict_mean: estimated from quantiles
- predict_median: direct from p50
- predict_mode: estimated from quantile density
- predict_quantile: interpolated from model quantiles
- predict_interval: prediction intervals from quantiles
- predict_std: estimated from IQR
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List


class MoiraiForecaster:
    """
    A simple scikit-learn-like interface for Moirai time series forecasting.
    
    Uses native quantile outputs from Moirai (p10, p20, ..., p90) to compute
    all distribution statistics directly.
    
    Parameters
    ----------
    model_name : str, default="Salesforce/moirai-2.0-R-small"
        Name of the pretrained Moirai model from HuggingFace.
        Options include:
        - "Salesforce/moirai-2.0-R-small"
        - "Salesforce/moirai-2.0-R-base"
        - "Salesforce/moirai-2.0-R-large"
        
    context_length : int, default=1000
        Number of past observations to use as context.
        
    device : str, default="auto"
        Device to run the model on ("auto", "cpu", "cuda").
    
    Examples
    --------
    >>> from moirai_sklearn import MoiraiForecaster
    >>> import numpy as np
    >>> 
    >>> ts = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
    >>> model = MoiraiForecaster()
    >>> predictions = model.predict(ts, horizon=30)
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/moirai-2.0-R-small",
        context_length: int = 1000,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.device = device
        
        self._model = None
        self._module = None
        self._quantile_levels = None
        
    def _ensure_model_loaded(self, horizon: int):
        """Lazy load the model on first use."""
        if self._model is None or self._model.hparams.prediction_length != horizon:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            
            if self._module is None:
                self._module = Moirai2Module.from_pretrained(self.model_name)
            
            self._model = Moirai2Forecast(
                module=self._module,
                prediction_length=horizon,
                context_length=self.context_length,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            
            if self.device != "auto":
                self._model = self._model.to(self.device)
                
            self._quantile_levels = np.array(self._module.quantile_levels)
    
    def _prepare_input(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, pd.Series):
            arr = data.values
        elif isinstance(data, np.ndarray):
            arr = data.flatten()
        elif isinstance(data, pd.DataFrame):
            # Use first column
            arr = data.iloc[:, 0].values
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return arr.astype(np.float32)
    
    def _raw_predict(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        horizon: int,
    ) -> np.ndarray:
        """
        Get raw quantile predictions from the model.
        
        Returns
        -------
        predictions : np.ndarray of shape (num_quantiles, horizon)
            Quantile predictions at each future time step.
        """
        self._ensure_model_loaded(horizon)
        arr = self._prepare_input(data)
        
        # Model.predict takes a list of arrays
        # Returns shape: (batch=1, num_quantiles, horizon)
        preds = self._model.predict(past_target=[arr])
        
        # Return shape: (num_quantiles, horizon)
        return preds[0]
    
    def predict(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        horizon: int,
    ) -> np.ndarray:
        """
        Predict future values (returns median/p50).
        
        Parameters
        ----------
        data : array-like
            Historical time series data.
        horizon : int
            Number of steps to forecast.
            
        Returns
        -------
        predictions : np.ndarray of shape (horizon,)
            Point predictions (median).
        """
        return self.predict_median(data, horizon)
    
    def predict_median(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        horizon: int,
    ) -> np.ndarray:
        """
        Predict median (p50) directly from model quantiles.
        """
        preds = self._raw_predict(data, horizon)
        median_idx = np.argmin(np.abs(self._quantile_levels - 0.5))
        return preds[median_idx]
    
    def predict_mean(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        horizon: int,
    ) -> np.ndarray:
        """
        Estimate mean as average of all quantiles.
        
        Simple approximation: mean ≈ average of [p10, p20, ..., p90]
        """
        preds = self._raw_predict(data, horizon)
        # Take mean across all quantiles
        return np.mean(preds, axis=0)
    
    def predict_mode(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        horizon: int,
    ) -> np.ndarray:
        """
        Estimate mode as the median (p50).
        
        For symmetric distributions, mode ≈ median.
        """
        return self.predict_median(data, horizon)
    
    def predict_quantile(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        horizon: int,
        q: Union[float, List[float], np.ndarray] = 0.5,
    ) -> np.ndarray:
        """
        Predict quantile(s) from model outputs.
        
        Parameters
        ----------
        q : float or array-like
            Quantile(s) to get. Must be one of the model's quantile levels:
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
        Returns
        -------
        quantiles : np.ndarray
            Shape (horizon,) if q is scalar, (horizon, n_quantiles) if q is array.
        """
        preds = self._raw_predict(data, horizon)
        q_arr = np.atleast_1d(q)
        
        # Find indices for requested quantiles
        results = np.zeros((horizon, len(q_arr)))
        for i, q_val in enumerate(q_arr):
            idx = np.argmin(np.abs(self._quantile_levels - q_val))
            if np.abs(self._quantile_levels[idx] - q_val) > 0.01:
                raise ValueError(
                    f"Quantile {q_val} not available. "
                    f"Available quantiles: {self._quantile_levels.tolist()}"
                )
            results[:, i] = preds[idx]
        
        if np.isscalar(q):
            return results[:, 0]
        return results
    
    def predict_interval(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        horizon: int,
        confidence: float = 0.8,
    ) -> np.ndarray:
        """
        Predict prediction interval from quantiles.
        
        Parameters
        ----------
        confidence : float, default=0.8
            Confidence level. Only 0.8 (p10-p90) and 0.6 (p20-p80) are available
            based on the model's quantile outputs.
        
        Returns
        -------
        intervals : np.ndarray of shape (horizon, 2)
            Lower and upper bounds [low, high].
        """
        if confidence == 0.8:
            return self.predict_quantile(data, horizon, q=[0.1, 0.9])
        elif confidence == 0.6:
            return self.predict_quantile(data, horizon, q=[0.2, 0.8])
        else:
            raise ValueError(
                f"Confidence level {confidence} not supported. "
                "Available: 0.8 (p10-p90) or 0.6 (p20-p80)"
            )
    
    def predict_std(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        horizon: int,
    ) -> np.ndarray:
        """
        Estimate standard deviation from p10-p90 range.
        
        For normal distribution: std ≈ (p90 - p10) / 2.56
        """
        preds = self._raw_predict(data, horizon)
        # p10 is at index 0, p90 is at index 8
        return (preds[8] - preds[0]) / 2.56
    
    def predict_all(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        horizon: int,
    ) -> pd.DataFrame:
        """
        Get all predictions in a DataFrame.
        
        Returns
        -------
        df : pd.DataFrame
            Columns: step, mean, median, std, p10, p20, ..., p90
        """
        preds = self._raw_predict(data, horizon)
        
        result = {
            'step': np.arange(1, horizon + 1),
            'mean': self.predict_mean(data, horizon),
            'median': self.predict_median(data, horizon),
            'std': self.predict_std(data, horizon),
        }
        
        # Add all model quantiles
        for i, q in enumerate(self._quantile_levels):
            result[f'p{int(q*100)}'] = preds[i]
        
        return pd.DataFrame(result)
