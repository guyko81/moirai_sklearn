# Contributing to moirai_sklearn

Thank you for your interest in contributing! Here's how you can help:

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/guyko81/moirai_sklearn.git
cd moirai_sklearn
```

2. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=moirai_sklearn
```

## Code Style

- Follow PEP 8 guidelines
- Add docstrings to all public methods
- Include type hints where appropriate

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Reporting Issues

Please use the GitHub issue tracker to report bugs or suggest features. Include:
- A clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Python version and environment details

## Questions?

Feel free to open an issue for questions, or discussions!
