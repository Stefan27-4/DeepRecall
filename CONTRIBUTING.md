# Contributing to DeepRecall

Thank you for your interest in contributing to DeepRecall! This guide will help you get started.

## Development Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Stefan27-4/DeepRecall.git
   cd DeepRecall
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. Install the package in editable mode with dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

We use **pytest** for testing:

```bash
pytest
```

To run with verbose output and coverage:

```bash
pytest -v --tb=short
```

## Code Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints for all public function signatures.
- Keep functions focused and small.
- Write docstrings for public modules, classes, and functions.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes in small, focused commits.
3. Add or update tests for any changed functionality.
4. Ensure all tests pass before submitting.
5. Open a pull request with a clear description of the changes and the motivation behind them.
6. Address any review feedback promptly.

## Reporting Issues

Please use the GitHub issue tracker to report bugs or request features. When filing a bug report, include:

- Steps to reproduce the issue
- Expected vs. actual behavior
- Python version and OS
- Relevant logs or error messages

See the `.github/ISSUE_TEMPLATE/` directory for issue templates, if available.
