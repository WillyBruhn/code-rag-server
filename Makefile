.PHONY: install run test test-cov lint format clean build dev-install update-index dev check fix_lint

# Development setup
install:
	pip install uv
	uv venv
	. .venv/bin/activate && uv pip install -e ".[dev]"

# Run the server
run:
	uv run code-rag-server

# Run tests without coverage
test:
	uv run pytest tests/

# Run tests with coverage
test-cov:
	uv pip install pytest-cov
	uv run pytest \
		--cov=code_rag_server \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-fail-under=70 \
		tests/

# Code quality
lint:
	uv run ruff check .
	uv run mypy .

# Format code
format:
	uv run ruff format .

# Fix linting issues
fix_lint:
	uv run ruff check --fix .

# Clean build artifacts
clean:
	rm -rf build/ dist/ .ruff_cache/ .mypy_cache/ **/__pycache__/ htmlcov/ .coverage

# Build package
build:
	uv pip build

# Update embeddings index
update-index:
	uv run code-rag-server --update-index

# Start development mode with hot reload
dev:
	uv run watchfiles "uv run code-rag-server" src/

# Install in development mode
dev-install:
	uv pip install -e .

# All quality checks
check: format lint test-cov