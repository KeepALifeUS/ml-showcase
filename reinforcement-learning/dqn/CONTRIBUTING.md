# Contributing to ML-DQN

Thank you for your interest in contributing to ML-DQN!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/KeepALifeUS/ml-dqn.git
cd ml-dqn

# Create virtual environment
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

```bash
# Format code
black src tests
isort src tests

# Type check
mypy src
```

## Testing

- All new features must have tests
- Maintain test coverage above 80%
- Use pytest markers for test categorization:
  - `@pytest.mark.unit` - Unit tests
  - `@pytest.mark.integration` - Integration tests
  - `@pytest.mark.slow` - Slow tests (skipped by default)

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run with coverage
pytest --cov=src --cov-report=html
```

## Adding New DQN Variants

1. Create a new class in `src/ml_dqn/agents/`
2. Inherit from `BaseDQN` or appropriate base class
3. Implement required methods:
   - `select_action()`
   - `learn()`
   - `_compute_loss()`
4. Add tests in `tests/test_agents/`
5. Document in README.md

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes following our code style
4. Add/update tests
5. Run the full test suite
6. Submit a PR with:
   - Clear description
   - Test results
   - Performance benchmarks (if applicable)

## Performance Benchmarks

When adding new algorithms, include benchmark results:

- CartPole-v1 (baseline)
- LunarLander-v2 (medium difficulty)
- Custom crypto environment (if applicable)

## Questions?

Open an issue or reach out to maintainers.
