# Contributing to Code RAG Server

First off, thank you for considering contributing to Code RAG Server! It's people like you that make it a great tool for everyone.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct: be respectful, inclusive, and professional in your interactions with other contributors.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/code-rag-server.git
   cd code-rag-server
   ```
3. Set up your development environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"  # Installs package in editable mode with development dependencies
   ```

## Development Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Format and lint your code:
   ```bash
   # Format code
   black .
   isort .
   
   # Run linters
   ruff check .
   mypy .
   ```
4. Run tests and ensure they pass:
   ```bash
   # Run tests with coverage report
   pytest
   
   # Run specific test file
   pytest tests/test_vector_store.py
   
   # Run tests with specific marker
   pytest -m "asyncio"
   ```
5. Update documentation as needed
6. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```
7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Create a Pull Request

## Testing

### Running Tests

The project uses pytest for testing. To run the tests:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov

# Run specific test file
pytest tests/test_vector_store.py

# Run tests matching specific name pattern
pytest -k "test_search"

# Run tests with specific marker
pytest -m "asyncio"
```

### Writing Tests

- Write tests for all new features
- Follow the existing test structure
- Use fixtures from conftest.py where possible
- Mock external services and file system operations
- Include both success and error cases
- Aim for high test coverage (minimum 80%)

Example test:
```python
@pytest.mark.asyncio
async def test_feature():
    # Arrange
    input_data = "test"
    
    # Act
    result = await process_data(input_data)
    
    # Assert
    assert result == expected_output
```

### Test Coverage

We aim for a minimum of 80% test coverage. Coverage reports are generated automatically when running tests. View the HTML report in `htmlcov/index.html`.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use type hints for function parameters and return values
- Write docstrings for public functions/methods/classes
- Maximum line length is 88 characters (Black default)
- Use descriptive variable names
- Keep functions focused and relatively small

### Code Formatting and Linting

The project uses several tools to ensure code quality:

```bash
# Format code
black .          # Code formatting
isort .          # Import sorting

# Lint code
ruff check .     # Fast Python linter
mypy .           # Type checking
```

### Example

```python
from pathlib import Path
from typing import List

def process_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Process files with specified extensions in a directory.

    Args:
        directory: The directory to process
        extensions: List of file extensions to include

    Returns:
        List of processed file paths
    """
    return list(
        path for path in directory.rglob("*")
        if path.suffix in extensions
    )
```

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes (formatting, missing semi colons, etc)
- refactor: Code refactoring
- test: Adding missing tests
- chore: Maintenance tasks

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Comment complex logic
- Keep documentation up to date with changes

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the version numbers following [Semantic Versioning](https://semver.org/)
3. Include tests for new functionality
4. Ensure all tests pass and code quality checks succeed
5. Link any related issues
6. The PR will be merged once you have the sign-off of a maintainer

## Getting Help

If you need help with anything:

1. Check existing issues and discussions
2. Open a new issue with your question
3. Reach out to maintainers

## Recognition

Contributors will be recognized in the project's documentation. We appreciate all contributions, big or small!