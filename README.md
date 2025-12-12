# moirai_sklearn

[![PyPI version](https://badge.fury.io/py/moirai_sklearn.svg)](https://badge.fury.io/py/moirai_sklearn)
[![Tests](https://github.com/guyko81/moirai_sklearn/workflows/Tests/badge.svg)](https://github.com/guyko81/moirai_sklearn/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A scikit-learn-like interface for Moirai time series forecasting.

Uses the native quantile predictions from Moirai (p10, p20, ..., p90) to compute distribution statistics directly - no artificial binning needed.

## Installation

```bash
pip install moirai_sklearn
```

Or from source:

```bash
git clone https://github.com/guyko81/moirai_sklearn.git
cd moirai_sklearn
pip install -e .
```

## Quick Start

```python
from moirai_sklearn import MoiraiForecaster
import numpy as np

ts = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1

model = MoiraiForecaster()
predictions = model.predict(ts, horizon=30)
intervals = model.predict_interval(ts, horizon=30, confidence=0.8)
```

## Methods

| Method | Description |
|--------|-------------|
| `predict(data, horizon)` | Point predictions (median) |
| `predict_mean(data, horizon)` | Mean estimated from quantiles |
| `predict_median(data, horizon)` | Median (p50) directly |
| `predict_mode(data, horizon)` | Mode from quantile density |
| `predict_quantile(data, horizon, q)` | Any quantile(s) via interpolation |
| `predict_interval(data, horizon, confidence)` | Prediction intervals |
| `predict_std(data, horizon)` | Std estimated from IQR |
| `predict_all(data, horizon)` | DataFrame with everything |

## Input Formats

```python
# NumPy array
model.predict(np.array([1, 2, 3, 4, 5]), horizon=10)

# Pandas Series
model.predict(pd.Series([1, 2, 3, 4, 5]), horizon=10)

# Pandas DataFrame (single column)
model.predict(pd.DataFrame({'value': [1, 2, 3, 4, 5]}), horizon=10)
```

## Models

- `Salesforce/moirai-2.0-R-small` (default)
- `Salesforce/moirai-2.0-R-base`
- `Salesforce/moirai-2.0-R-large`

## Features

✨ **Clean API**: Simple, intuitive scikit-learn-style interface  
📊 **Rich Statistics**: Mean, median, mode, quantiles, intervals, and standard deviation  
🎯 **Native Quantiles**: Uses Moirai's built-in quantile predictions (no binning)  
🔌 **Flexible Input**: Supports NumPy arrays, Pandas Series, and DataFrames  
⚡ **Efficient**: Smart model caching for repeated predictions  

## Examples

Check out the `examples/basic_usage.py` file for more detailed examples including:
- Simple forecasting
- Prediction intervals
- Multiple quantiles
- Pandas DataFrame inputs
- Visualization

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this package in your research, please cite the original Moirai paper:

```bibtex
@article{woo2024unified,
  title={Unified Training of Universal Time Series Forecasting Transformers},
  author={Woo, Gerald and Liu, Chenghao and Kumar, Akshat and Xiong, Caiming and Savarese, Silvio and Sahoo, Doyen},
  journal={arXiv preprint arXiv:2402.02592},
  year={2024}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) file for details.

## Acknowledgments

This package is a wrapper around [uni2ts](https://github.com/SalesforceAIResearch/uni2ts), the official implementation of Moirai by Salesforce AI Research.
