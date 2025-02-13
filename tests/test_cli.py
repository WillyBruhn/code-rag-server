from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from code_rag_server.cli import _index_repository, cli

BASE_PATH = Path(__file__).resolve().parent.parent

@pytest.fixture
def mock_metadata():
    """Create mock metadata for testing."""
    return {
        "file": "test.py",
        "repo": "test-repo",
        "code": "def test(): pass",
        "language": "python",
    }

@pytest.fixture
def mock_vector_store(mock_metadata):
    """Create a mock vector store."""
    mock_store = MagicMock()
    mock_store.size = 1
    mock_store.metadata = [mock_metadata]
    mock_store.search.return_value = [(mock_metadata, 0.95)]
    mock_store.embeddings = [[0.1] * 768]

    with patch("code_rag_server.vector_store.InMemoryVectorStore") as mock_class:
        mock_class.return_value = mock_store
        yield mock_class

@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock_embed = AsyncMock()
    mock_embed.get_embedding = AsyncMock(return_value=[0.1] * 768)
    mock_embed.get_batch_embeddings = AsyncMock(return_value=[[0.1] * 768])

    with patch("code_rag_server.embeddings.EmbeddingService") as mock_class:
        mock_class.return_value = mock_embed
        yield mock_class

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()

def test_cli_no_args(cli_runner):
    """Test CLI with no arguments."""
    result = cli_runner.invoke(cli)
    assert result.exit_code == 0
    assert "Usage:" in result.output

@pytest.mark.run_with_lm_studio
def test_index_command(cli_runner, tmp_path, mock_vector_store, mock_embedding_service):
    """Test the index command."""
    # Create a test repository
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()
    (repo_path / "test.py").write_text("def test(): pass")

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.relative_to", return_value=Path("test-repo")),
        patch("pathlib.Path.rglob") as mock_rglob
    ):
        mock_rglob.return_value = [repo_path / "test.py"]
        result = cli_runner.invoke(cli, ["index", str(repo_path)])
        assert result.exit_code == 0
        assert "Starting indexing process" in result.output

def test_index_nonexistent_path(cli_runner):
    """Test indexing a nonexistent path."""
    result = cli_runner.invoke(cli, ["index", "/nonexistent/path"])
    assert result.exit_code == 2  # Click's error exit code
    assert "Error:" in result.output

def test_search_command(cli_runner, mock_metadata):
    """Test the search command."""
    # Create mock store with specific search result
    mock_store = MagicMock()
    mock_store.size = 1
    mock_store.search.return_value = [(mock_metadata, 0.95)]
    mock_store.metadata = [mock_metadata]

    # Create mock embedding service
    mock_embed = AsyncMock()
    mock_embed.get_embedding = AsyncMock(return_value=[0.1] * 768)

    print("\nDebug info:")
    print("Mock metadata:", mock_metadata)
    print("Mock search result:", mock_store.search.return_value)

    # Mock both services directly in the CLI module
    with (
        patch("code_rag_server.cli.InMemoryVectorStore", return_value=mock_store),
        patch("code_rag_server.cli.EmbeddingService", return_value=mock_embed),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.glob") as mock_glob
    ):
        mock_glob.return_value = [Path(f"{BASE_PATH}/indices/test-repo.pkl")]

        result = cli_runner.invoke(cli, ["search", "test function"])

        print("\nSearch output:", result.output)
        print("Search mock called:", mock_store.search.call_count, "times")
        if mock_store.search.call_count > 0:
            print("Search args:", mock_store.search.call_args)

        assert result.exit_code == 0
        assert "File: test.py" in result.output
        assert "Repository: test-repo" in result.output
        assert "Score: 0.95" in result.output
        assert "def test(): pass" in result.output

        # Verify mocks were called correctly
        mock_store.search.assert_called_once()

def test_invalid_num_results(cli_runner):
    """Test search command with invalid number of results."""
    result = cli_runner.invoke(cli, ["search", "test", "-n", "-1"])
    assert result.exit_code == 2  # Click's error exit code
    assert "Error:" in result.output

def test_search_no_index(cli_runner):
    """Test search when no index exists."""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.glob") as mock_glob
    ):
        mock_glob.return_value = []

        result = cli_runner.invoke(cli, ["search", "test"])
        assert result.exit_code == 2
        assert "No indexes found" in result.output

def test_help_command(cli_runner):
    """Test the help command."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Options:" in result.output

@pytest.mark.asyncio
async def test_index_repository(tmp_path, mock_vector_store, mock_embedding_service):
    """Test the _index_repository function."""
    # Create a test repository
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()
    (repo_path / "test.py").write_text("def test(): pass")

    with patch("code_rag_server.vector_store.InMemoryVectorStore", mock_vector_store):
        mock_rglob = MagicMock()
        mock_rglob.return_value = [repo_path / "test.py"]
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.relative_to", return_value=Path("test-repo")),
            patch("pathlib.Path.rglob", mock_rglob),
            patch("httpx.AsyncClient.post") as mock_post
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"data": [{"embedding": [0.1] * 768}]})
            )
            await _index_repository(repo_path)
