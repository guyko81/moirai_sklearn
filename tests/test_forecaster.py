"""
Tests for MoiraiForecaster
"""

import numpy as np
import pandas as pd
import pytest
from moirai_sklearn import MoiraiForecaster


@pytest.fixture
def simple_timeseries():
    """Create a simple sine wave time series for testing."""
    return np.sin(np.linspace(0, 10, 100)) + np.random.RandomState(42).randn(100) * 0.1


@pytest.fixture
def forecaster():
    """Create a forecaster instance."""
    return MoiraiForecaster(model_name="Salesforce/moirai-2.0-R-small")


class TestMoiraiForecaster:
    
    def test_init(self):
        """Test forecaster initialization."""
        model = MoiraiForecaster()
        assert model.model_name == "Salesforce/moirai-2.0-R-small"
        assert model.context_length == 1000
        assert model.device == "auto"
    
    def test_predict_with_numpy(self, forecaster, simple_timeseries):
        """Test predict with numpy array input."""
        predictions = forecaster.predict(simple_timeseries, horizon=10)
        assert predictions.shape == (10,)
        assert not np.isnan(predictions).any()
    
    def test_predict_with_pandas_series(self, forecaster, simple_timeseries):
        """Test predict with pandas Series input."""
        series = pd.Series(simple_timeseries)
        predictions = forecaster.predict(series, horizon=10)
        assert predictions.shape == (10,)
        assert not np.isnan(predictions).any()
    
    def test_predict_with_dataframe(self, forecaster, simple_timeseries):
        """Test predict with pandas DataFrame input."""
        df = pd.DataFrame({'value': simple_timeseries})
        predictions = forecaster.predict(df, horizon=10)
        assert predictions.shape == (10,)
        assert not np.isnan(predictions).any()
    
    def test_predict_median(self, forecaster, simple_timeseries):
        """Test predict_median."""
        median = forecaster.predict_median(simple_timeseries, horizon=10)
        assert median.shape == (10,)
        assert not np.isnan(median).any()
    
    def test_predict_mean(self, forecaster, simple_timeseries):
        """Test predict_mean."""
        mean = forecaster.predict_mean(simple_timeseries, horizon=10)
        assert mean.shape == (10,)
        assert not np.isnan(mean).any()
    
    def test_predict_mode(self, forecaster, simple_timeseries):
        """Test predict_mode."""
        mode = forecaster.predict_mode(simple_timeseries, horizon=10)
        assert mode.shape == (10,)
        assert not np.isnan(mode).any()
    
    def test_predict_quantile_single(self, forecaster, simple_timeseries):
        """Test predict_quantile with single quantile."""
        q50 = forecaster.predict_quantile(simple_timeseries, horizon=10, q=0.5)
        assert q50.shape == (10,)
        assert not np.isnan(q50).any()
    
    def test_predict_quantile_multiple(self, forecaster, simple_timeseries):
        """Test predict_quantile with multiple quantiles."""
        quantiles = forecaster.predict_quantile(simple_timeseries, horizon=10, q=[0.1, 0.5, 0.9])
        assert quantiles.shape == (10, 3)
        assert not np.isnan(quantiles).any()
        # Check ordering: q10 <= q50 <= q90
        assert (quantiles[:, 0] <= quantiles[:, 1]).all()
        assert (quantiles[:, 1] <= quantiles[:, 2]).all()
    
    def test_predict_quantile_invalid(self, forecaster, simple_timeseries):
        """Test predict_quantile with invalid quantile raises error."""
        with pytest.raises(ValueError, match="Quantile.*not available"):
            forecaster.predict_quantile(simple_timeseries, horizon=10, q=0.99)
    
    def test_predict_interval_80(self, forecaster, simple_timeseries):
        """Test predict_interval with 80% confidence."""
        intervals = forecaster.predict_interval(simple_timeseries, horizon=10, confidence=0.8)
        assert intervals.shape == (10, 2)
        assert not np.isnan(intervals).any()
        # Check lower <= upper
        assert (intervals[:, 0] <= intervals[:, 1]).all()
    
    def test_predict_interval_60(self, forecaster, simple_timeseries):
        """Test predict_interval with 60% confidence."""
        intervals = forecaster.predict_interval(simple_timeseries, horizon=10, confidence=0.6)
        assert intervals.shape == (10, 2)
        assert not np.isnan(intervals).any()
        # Check lower <= upper
        assert (intervals[:, 0] <= intervals[:, 1]).all()
    
    def test_predict_interval_invalid(self, forecaster, simple_timeseries):
        """Test predict_interval with invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence level.*not supported"):
            forecaster.predict_interval(simple_timeseries, horizon=10, confidence=0.95)
    
    def test_predict_std(self, forecaster, simple_timeseries):
        """Test predict_std."""
        std = forecaster.predict_std(simple_timeseries, horizon=10)
        assert std.shape == (10,)
        assert not np.isnan(std).any()
        assert (std > 0).all()  # Standard deviation should be positive
    
    def test_predict_all(self, forecaster, simple_timeseries):
        """Test predict_all returns comprehensive DataFrame."""
        df = forecaster.predict_all(simple_timeseries, horizon=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        
        # Check expected columns
        expected_cols = ['step', 'mean', 'median', 'std', 
                        'p10', 'p20', 'p30', 'p40', 'p50', 
                        'p60', 'p70', 'p80', 'p90']
        assert all(col in df.columns for col in expected_cols)
        
        # Check no NaN values
        assert not df.isnull().any().any()
        
        # Check quantile ordering
        assert (df['p10'] <= df['p50']).all()
        assert (df['p50'] <= df['p90']).all()
    
    def test_different_horizons(self, forecaster, simple_timeseries):
        """Test prediction with different horizons."""
        for horizon in [5, 10, 20, 50]:
            predictions = forecaster.predict(simple_timeseries, horizon=horizon)
            assert predictions.shape == (horizon,)
            assert not np.isnan(predictions).any()
    
    def test_model_caching(self, simple_timeseries):
        """Test that model is cached and reused."""
        forecaster = MoiraiForecaster()
        
        # First prediction - loads model
        _ = forecaster.predict(simple_timeseries, horizon=10)
        assert forecaster._model is not None
        assert forecaster._module is not None
        
        # Second prediction - reuses model
        model_id = id(forecaster._module)
        _ = forecaster.predict(simple_timeseries, horizon=10)
        assert id(forecaster._module) == model_id  # Same module instance
    
    def test_invalid_input_type(self, forecaster):
        """Test that invalid input type raises error."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            forecaster.predict([1, 2, 3, 4, 5], horizon=10)  # List not supported


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
