import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag_server.server import (
    _clone_github_repo,
    _guess_language,
    _should_ignore,
    _validate_github_url,
    handle_call_tool,
    handle_list_tools,
    update_index,
)

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
def mock_embedding_result():
    """Create a mock embedding result."""
    return [0.1] * 768  # Match embedding dimension


@pytest.fixture
def mock_vector_store(mock_metadata):
    """Create a mock vector store."""
    with patch("code_rag_server.vector_store.InMemoryVectorStore") as mock_class:
        instance = MagicMock()
        instance.size = 1
        instance.metadata = [mock_metadata]
        instance.search.return_value = [(mock_metadata, 0.95)]
        instance.embeddings = [[0.1] * 768]
        mock_class.return_value = instance
        yield mock_class


# Test GitHub URL validation
def test_validate_github_url_https():
    """Test validation of HTTPS GitHub URLs."""
    owner, repo = _validate_github_url("https://github.com/owner/repo")
    assert owner == "owner"
    assert repo == "repo"


def test_validate_github_url_ssh():
    """Test validation of SSH GitHub URLs."""
    owner, repo = _validate_github_url("git@github.com:owner/repo.git")
    assert owner == "owner"
    assert repo == "repo"


def test_validate_github_url_invalid():
    """Test validation of invalid GitHub URLs."""
    with pytest.raises(ValueError):
        _validate_github_url("invalid_url")


# Test repository cloning
@pytest.mark.asyncio
async def test_clone_github_repo(tmp_path):
    """Test GitHub repository cloning."""
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.parent") as mock_parent,
    ):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Cloning into 'repo'...", stderr=""
        )
        mock_parent.mkdir = MagicMock()

        repo_path = _clone_github_repo("https://github.com/owner/repo")
        assert isinstance(repo_path, Path)
        assert "owner" in str(repo_path)
        assert "repo" in str(repo_path)


@pytest.mark.asyncio
async def test_clone_github_repo_with_branch(tmp_path):
    """Test GitHub repository cloning with specific branch."""
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.parent") as mock_parent,
    ):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Cloning into 'repo'...", stderr=""
        )
        mock_parent.mkdir = MagicMock()

        repo_path = _clone_github_repo("https://github.com/owner/repo", branch="dev")
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert "-b" in cmd_args
        assert "dev" in cmd_args
        assert isinstance(repo_path, Path)


@pytest.mark.asyncio
async def test_clone_github_repo_error():
    """Test handling of GitHub clone errors."""
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.exists", return_value=False),
    ):
        mock_run.side_effect = subprocess.CalledProcessError(
            128, "git clone", stderr=b"Repository not found"
        )

        with pytest.raises(subprocess.CalledProcessError):
            _clone_github_repo("https://github.com/owner/nonexistent")


# Test file operations
def test_should_ignore():
    """Test file ignore patterns."""
    assert _should_ignore(Path(".git/config"))
    assert _should_ignore(Path("node_modules/package.json"))
    assert _should_ignore(Path("venv/lib/python3.8"))
    assert not _should_ignore(Path("src/main.py"))
    assert not _should_ignore(Path("README.md"))


def test_guess_language():
    """Test programming language detection."""
    assert _guess_language(Path("test.py")) == "python"
    assert _guess_language(Path("main.js")) == "javascript"
    assert _guess_language(Path("style.css")) == "css"
    assert _guess_language(Path("unknown.xyz")) == ""


# Test MCP server functionality
@pytest.mark.asyncio
async def test_handle_list_tools():
    """Test listing available tools."""
    tools = await handle_list_tools()

    assert isinstance(tools, list)
    tool_names = {tool.name for tool in tools}
    assert "clone_repo" in tool_names
    assert "search_code" in tool_names
    assert "get_file" in tool_names
    assert "update_index" in tool_names


@pytest.mark.asyncio
async def test_handle_call_tool_clone_repo(tmp_path):
    """Test clone_repo tool handling."""
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.parent") as mock_parent,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        mock_parent.mkdir = MagicMock()

        result = await handle_call_tool(
            "clone_repo", {"repository_url": "https://github.com/owner/repo"}
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert "Successfully cloned" in result[0].text


@pytest.mark.asyncio
async def test_handle_call_tool_update_index(tmp_path, mock_metadata):
    """Test update_index tool handling."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def test(): pass")

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.relative_to", return_value=Path("test-repo")),
        patch("pathlib.Path.rglob") as mock_rglob,
        patch("code_rag_server.vector_store.InMemoryVectorStore") as MockStore,
        patch(
            "code_rag_server.embeddings.EmbeddingService.get_batch_embeddings",
            new_callable=AsyncMock,
        ) as mock_embed,
    ):
        mock_rglob.return_value = [test_file]
        mock_store = MockStore.return_value
        mock_store.size = 0  # Indicate no existing index
        mock_embed.return_value = [[0.1] * 768]

        result = await handle_call_tool("update_index", {"repo_path": str(tmp_path)})

        assert isinstance(result, list)
        assert len(result) == 1
        assert "Successfully indexed" in result[0].text


@pytest.mark.asyncio
async def test_handle_call_tool_invalid():
    """Test handling of invalid tool calls."""
    with pytest.raises(ValueError):
        await handle_call_tool("invalid_tool", {})


@pytest.mark.asyncio
async def test_update_index(tmp_path, mock_metadata):
    """Test repository indexing functionality."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def test(): pass")

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.relative_to", return_value=Path("test-repo")),
        patch("pathlib.Path.rglob") as mock_rglob,
        patch("code_rag_server.vector_store.InMemoryVectorStore") as MockStore,
        patch(
            "code_rag_server.embeddings.EmbeddingService.get_batch_embeddings",
            new_callable=AsyncMock,
        ) as mock_embed,
    ):
        mock_rglob.return_value = [test_file]
        mock_store = MockStore.return_value
        mock_store.size = 0  # Indicate no existing index
        mock_embed.return_value = [[0.1] * 768]

        result = await update_index(tmp_path)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "Successfully indexed" in result[0].text


@pytest.mark.asyncio
async def test_update_index_nonexistent_path():
    """Test indexing with nonexistent repository path."""
    with patch("pathlib.Path.exists", return_value=False):
        result = await update_index(Path("/nonexistent/path"))
        assert "path does not exist" in result[0].text
