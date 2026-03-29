#!/bin/bash

# Lint script using mypy and ruff

set -e

echo "Running ruff..."
uv run ruff format .
uv run ruff check . --fix

echo "Running ty..."
uv run ty check --error-on-warning .

echo "Linting complete!"