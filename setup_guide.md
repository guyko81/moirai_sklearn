# Setup Guide for moirai_sklearn Repository

## Initial Git Setup

Once you've moved this folder to its own location, initialize it as a git repository:

```bash
# Navigate to the moirai_sklearn folder
cd path/to/moirai_sklearn

# Initialize git repository
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial commit: moirai_sklearn package"

# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/guyko81/moirai_sklearn.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Update pyproject.toml

Before publishing, update the author information in `pyproject.toml`:

```toml
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
```

## Local Development

Install the package in development mode:

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=moirai_sklearn --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

## Publishing to PyPI

### 1. Test on TestPyPI first

```bash
# Build the package
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ moirai_sklearn
```

### 2. Publish to PyPI

```bash
# Build the package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

### 3. Automatic Publishing (Recommended)

The repository includes a GitHub Actions workflow (`.github/workflows/publish.yml`) that automatically publishes to PyPI when you create a release:

1. Add your PyPI API token as a GitHub secret named `PYPI_API_TOKEN`
2. Create a new release on GitHub
3. The package will be automatically built and published

## Repository Structure

```
moirai_sklearn/
├── moirai_sklearn/          # Package source code
│   ├── __init__.py
│   └── forecaster.py
├── tests/                   # Test files
│   ├── __init__.py
│   └── test_forecaster.py
├── examples/                # Example scripts
│   └── basic_usage.py
├── .github/                 # GitHub Actions workflows
│   └── workflows/
│       ├── test.yml         # CI tests
│       └── publish.yml      # PyPI publishing
├── pyproject.toml          # Package configuration
├── README.md               # Main documentation
├── LICENSE                 # Apache 2.0 license
├── CONTRIBUTING.md         # Contribution guidelines
├── MANIFEST.in            # Include non-Python files
├── .gitignore             # Git ignore rules
└── setup_guide.md         # This file
```

## Next Steps

1. ✅ Move this folder out of uni2ts
2. ✅ Initialize git repository
3. ✅ Update author info in pyproject.toml
4. Run tests to ensure everything works
5. Create GitHub repository
6. Push code to GitHub
7. Set up PyPI account and get API token
8. Publish package!

## Questions?

Open an issue on GitHub or check the CONTRIBUTING.md file for more information.
