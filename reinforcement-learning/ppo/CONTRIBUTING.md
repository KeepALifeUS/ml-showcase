# Contributing to ML-PPO

Thank you for your interest in contributing to ML-PPO!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/KeepALifeUS/ml-ppo.git
cd ml-ppo

# Create virtual environment (Python 3.11+ required)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Code Style

We use the following tools for code quality:

- **Black** - Code formatting (line length: 120)
- **isort** - Import sorting
- **mypy** - Static type checking
- **flake8** - Linting

```bash
# Format code
black src tests
isort src tests

# Type check
mypy src

# Lint
flake8 src tests
```

## Testing

- All new features must have tests
- Use pytest markers for test categorization:
  - `@pytest.mark.unit` - Unit tests
  - `@pytest.mark.integration` - Integration tests
  - `@pytest.mark.slow` - Slow tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run with coverage
pytest --cov=src --cov-report=html
```

## Adding New PPO Variants

1. Create a new module in `src/core/`
2. Inherit from `BasePPO` or appropriate base class
3. Implement required methods:
   - `compute_surrogate_loss()`
   - `compute_value_loss()`
   - `update_policy()`
4. Add tests in `tests/`
5. Document in README.md

## Network Architectures

When adding new network architectures:

1. Create in `src/networks/`
2. Support both shared and separate actor-critic
3. Include initialization methods
4. Document input/output shapes

## Distributed Training

For distributed training contributions:

1. Ensure Ray compatibility
2. Test with multiple workers
3. Document scaling behavior
4. Include fault tolerance

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes following our code style
4. Add/update tests
5. Run the full test suite
6. Submit a PR with:
   - Clear description
   - Test results
   - Benchmark comparisons (if applicable)

## Performance Benchmarks

When adding new features, include benchmarks:

- CartPole-v1 (learning speed)
- LunarLander-v2 (final performance)
- Crypto environment (if applicable)

## Questions?

Open an issue or reach out to maintainers.
