import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def temp_repo_dir(temp_dir):
    """Create a temporary repository structure."""
    repo_dir = temp_dir / "test-repo"
    repo_dir.mkdir(parents=True)

    # Create some test files
    (repo_dir / "test.py").write_text("def test(): pass")
    (repo_dir / "README.md").write_text("# Test Repository")

    return repo_dir


@pytest.fixture
def mock_embedding_response():
    """Create a mock embedding API response."""
    return {
        "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}],
        "model": "text-embedding-nomic-embed-text-v1.5@q8_0",
    }


@pytest.fixture
def sample_code_files(temp_dir):
    """Create sample code files for testing."""
    files = {
        "main.py": "def main():\n    print('Hello')",
        "test.js": "function test() { return true; }",
        "style.css": "body { color: black; }",
        "README.md": "# Project",
    }

    for name, content in files.items():
        (temp_dir / name).write_text(content)

    return temp_dir


@pytest.fixture
def indices_dir(temp_dir):
    """Create a temporary indices directory."""
    indices = temp_dir / "indices"
    indices.mkdir()
    return indices
