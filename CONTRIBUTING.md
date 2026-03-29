# Contributing to primacore

Thank you for your interest in contributing to primacore! We welcome contributions from everyone. This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions with other contributors and maintainers.

## Getting Started

### Prerequisites

- Python >=3.10
- Git

### Setting up the Development Environment

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/yourusername/primacore.git
   cd primacore
   ```

2. Install the project in development mode using uv:
   ```bash
   uv sync
   ```

3. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests for new functionality.

3. Run the tests to ensure everything works:
   ```bash
   ./scripts/test.sh
   ```

4. Run linting to ensure code quality:
   ```bash
   ./scripts/lint.sh
   ```

5. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: description of changes"
   ```

6. Push to your fork and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Testing

- Write tests for all new features and bug fixes.
- Place tests in the `tests/` directory, mirroring the package structure.
- Run tests using: `./scripts/test.sh`

## Code Style

- The project uses **Black** for code formatting.
- The project uses **Ruff** for linting.
- Run `./scripts/lint.sh` to check and auto-fix code style issues.
- Use clear variable names and add comments for complex logic.

## Documentation

- Update documentation for any user-facing changes.
- Documentation is built using Sphinx and located in the `docs/` directory.
- Use docstrings in Google style format for all public functions and classes.

Example docstring:
```python
def reconstruct_climate(pollen_data: pd.DataFrame, method: str = "mat") -> pd.DataFrame:
    """Reconstruct paleoclimate from pollen data.
    
    Args:
        pollen_data: DataFrame containing pollen percentages
        method: Reconstruction method ('mat', 'brt', 'rf', 'wapls')
        
    Returns:
        DataFrame containing reconstructed climate variables
        
    Raises:
        ValueError: If method is not recognized
    """
```

## Pull Request Process

1. Ensure all tests pass: `./scripts/test.sh`
2. Ensure code is properly formatted: `./scripts/lint.sh`
3. Update documentation if needed
4. Create a clear PR description explaining:
   - What problem does this solve?
   - How was it solved?
   - What tests were added?
5. Respond to feedback and re-request review after making changes

## Reporting Issues

When reporting bugs, please include:
- A clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Python version and environment details
- Any relevant error messages or logs

## Questions?

Feel free to open an issue or start a discussion if you have questions about contributing.

Thank you for contributing to primacore!
