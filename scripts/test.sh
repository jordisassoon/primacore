#!/usr/bin/env bash

set -euo pipefail

echo "Running tests..."
uv run pytest --vcr-record=none

echo "All tests passed!"