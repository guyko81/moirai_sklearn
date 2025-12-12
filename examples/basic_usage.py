"""
Basic usage examples for moirai_sklearn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moirai_sklearn import MoiraiForecaster


def example_simple_forecast():
    """Simple forecasting example with a sine wave."""
    print("Example 1: Simple Forecast")
    print("-" * 50)
    
    # Create a simple time series (sine wave with noise)
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    ts = np.sin(t) + np.random.randn(100) * 0.1
    
    # Create forecaster and predict
    model = MoiraiForecaster()
    predictions = model.predict(ts, horizon=30)
    
    print(f"Input length: {len(ts)}")
    print(f"Forecast horizon: {len(predictions)}")
    print(f"First 5 predictions: {predictions[:5]}")
    print()


def example_with_intervals():
    """Example showing prediction intervals."""
    print("Example 2: Prediction with Intervals")
    print("-" * 50)
    
    # Create time series
    np.random.seed(42)
    ts = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
    
    # Get predictions with intervals
    model = MoiraiForecaster()
    median = model.predict_median(ts, horizon=20)
    intervals_80 = model.predict_interval(ts, horizon=20, confidence=0.8)
    
    print(f"Median prediction (first 5): {median[:5]}")
    print(f"80% interval (first 5):")
    print(f"  Lower: {intervals_80[:5, 0]}")
    print(f"  Upper: {intervals_80[:5, 1]}")
    print()


def example_all_statistics():
    """Example showing all available statistics."""
    print("Example 3: All Statistics")
    print("-" * 50)
    
    # Create time series
    np.random.seed(42)
    ts = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
    
    # Get all statistics
    model = MoiraiForecaster()
    df = model.predict_all(ts, horizon=10)
    
    print("Complete forecast DataFrame:")
    print(df.head())
    print()
    print("Available columns:", df.columns.tolist())
    print()


def example_pandas_input():
    """Example using pandas DataFrame as input."""
    print("Example 4: Pandas DataFrame Input")
    print("-" * 50)
    
    # Create pandas DataFrame
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': np.sin(np.linspace(0, 10, 100)) + np.random.RandomState(42).randn(100) * 0.1
    })
    
    # Forecast from DataFrame
    model = MoiraiForecaster()
    predictions = model.predict(df[['value']], horizon=15)
    
    print("Input DataFrame shape:", df.shape)
    print("Input DataFrame head:")
    print(df.head())
    print()
    print(f"Predictions (first 5): {predictions[:5]}")
    print()


def example_multiple_quantiles():
    """Example showing multiple quantile predictions."""
    print("Example 5: Multiple Quantiles")
    print("-" * 50)
    
    # Create time series
    np.random.seed(42)
    ts = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
    
    # Get multiple quantiles
    model = MoiraiForecaster()
    quantiles = model.predict_quantile(ts, horizon=10, q=[0.1, 0.3, 0.5, 0.7, 0.9])
    
    print("Quantile predictions shape:", quantiles.shape)
    print("Quantiles [0.1, 0.3, 0.5, 0.7, 0.9] for step 1:")
    print(quantiles[0])
    print()


def example_visualization():
    """Example with visualization."""
    print("Example 6: Visualization")
    print("-" * 50)
    
    # Create time series
    np.random.seed(42)
    ts = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
    
    # Get predictions
    model = MoiraiForecaster()
    horizon = 30
    median = model.predict_median(ts, horizon=horizon)
    intervals_80 = model.predict_interval(ts, horizon=horizon, confidence=0.8)
    intervals_60 = model.predict_interval(ts, horizon=horizon, confidence=0.6)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Historical data
    plt.plot(range(len(ts)), ts, label='Historical', color='black', linewidth=2)
    
    # Forecast
    forecast_x = range(len(ts), len(ts) + horizon)
    plt.plot(forecast_x, median, label='Forecast (median)', color='blue', linewidth=2)
    
    # Intervals
    plt.fill_between(forecast_x, intervals_80[:, 0], intervals_80[:, 1], 
                     alpha=0.3, color='blue', label='80% interval')
    plt.fill_between(forecast_x, intervals_60[:, 0], intervals_60[:, 1], 
                     alpha=0.5, color='blue', label='60% interval')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Moirai Forecast with Prediction Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save or show
    plt.tight_layout()
    plt.savefig('forecast_example.png', dpi=150)
    print("Visualization saved to 'forecast_example.png'")
    plt.close()


if __name__ == "__main__":
    # Run all examples
    example_simple_forecast()
    example_with_intervals()
    example_all_statistics()
    example_pandas_input()
    example_multiple_quantiles()
    
    # Visualization example (requires matplotlib)
    try:
        example_visualization()
    except ImportError:
        print("Skipping visualization example (matplotlib not installed)")
