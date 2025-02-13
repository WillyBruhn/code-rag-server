# Code RAG Server

Model Context Protocol (MCP) server that enables semantic code search through repositories using RAG (Retrieval Augmented Generation). It indexes your code repositories and provides natural language and code-based search capabilities, making it easier to find relevant code snippets and patterns across your codebase.

## Features

- 🔍 Semantic code search using natural language or code snippets
- 📁 File path based search
- 🤖 MCP server integration for AI assistants
- 📚 Repository indexing with embeddings
- 🔄 Compatible with Cline and Roo-Cline

## Prerequisites

- Python 3.10 or higher
- uv (Python package installer)
- Git (for cloning repositories)
- LM Studio with an embedding model (required for semantic search)

## Setting Up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Open LM Studio and go to the Models tab
3. Download the embedding model: `nomic-embed-text-v1.5`
4. Run the model in Embedding Server mode
   - The server runs on http://127.0.0.1:1235
   - The API endpoint should be available at http://127.0.0.1:1235/v1/embeddings
   - Test the endpoint with:
     ```bash
     curl http://127.0.0.1:1235/v1/embeddings \
       -H "Content-Type: application/json" \
       -d '{
         "model": "text-embedding-nomic-embed-text-v1.5@q8_0",
         "input": "Some text to embed"
       }'
     ```
   - Keep LM Studio running while using code-rag-server

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/code-rag-server.git
cd code-rag-server
```

2. Install using Make:
```bash
make install
```

Or manually with uv:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Development

### Setup Development Environment
```bash
# Install with development dependencies
make install
```

### Running Tests
```bash
# Run basic tests
make test

# Run tests with coverage report
make test-cov

# View coverage report
open htmlcov/index.html

# Run all quality checks (format, lint, test)
make check
```

### Code Quality
```bash
# Format code
make format

# Run linters
make lint

# Fix linting issues automatically
make fix_lint

# Note: Directories can be excluded from linting in pyproject.toml:
# - For ruff: Add directories to 'exclude' list under [tool.ruff]
#   Default excluded: .git, .venv, __pycache__, build, dist, github_repos, indices
# - For mypy: Add patterns to 'exclude' list under [tool.mypy]
#   Default excluded: github_repos/.*, build/.*, dist/.*
```

## Integration with Cline/Roo-Cline

To use code-rag-server with Cline or Roo-Cline, add the following configuration to your MCP settings file (`cline_mcp_settings.json`):

```json
{
  "mcpServers": {
    "code-rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/code-rag-server",
        "run",
        "code-rag-server"
      ],
      "disabled": false,
      "alwaysAllow": [],
      "timeout": 300
    }
  }
}
```

Replace `/path/to/code-rag-server` with the actual path to your code-rag-server installation.

The server provides the following MCP tools:
- `search_code`: Semantic search through indexed repositories
- `get_file`: Retrieve specific file contents
- `update_index`: Index new repositories

## Usage

### Command Line Interface

#### Indexing a Repository

To index a repository for searching:

```bash
python -m code_rag_server.cli index /path/to/repository
```

The embeddings will be stored in the `indices` directory with a structure that matches the repository path relative to the `github_repos` directory.

#### Searching Code

There are two search modes available:

1. **Semantic Search** (default):
```bash
python -m code_rag_server.cli search "your search query" -n 5
```
You can use either natural language queries or code snippets to find similar code.

2. **File Search** (search by file path):
```bash
python -m code_rag_server.cli search "filename_or_path" --file-mode -n 5
```

Options:
- `-n`, `--num-results`: Number of results to return (default: 5)
- `--file-mode`: Search for files by path
- `--no-file-mode`: Use semantic search (default)

### MCP Server

The tool provides an MCP server interface with the following tools:

#### search_code
Semantic search through indexed code repositories:
```json
{
    "tool": "search_code",
    "args": {
        "query": "your search query or code snippet",
        "num_results": 5,
        "repository": "semantic-kernel/python"  // Optional: limit search to specific repository
    }
}
```

#### get_file
Get the exact content of a specific file:
```json
{
    "tool": "get_file",
    "args": {
        "file_path": "tests/utils.py",
        "repository": "semantic-kernel/python"
    }
}
```

#### update_index
Index a repository for searching:
```json
{
    "tool": "update_index",
    "args": {
        "repo_path": "/path/to/repository"
    }
}
```

## Examples

### Command Line Examples

Index a repository:
```bash
python -m code_rag_server.cli index /path/to/github_repos/semantic-kernel/python
```

Search for code semantically:
```bash
# Natural language query
python -m code_rag_server.cli search "how to initialize an agent" -n 3

# Code snippet query
python -m code_rag_server.cli search "async def update_index(repo_path: Path):" -n 3
```

Search for specific files:
```bash
python -m code_rag_server.cli search "test_bedrock_agent.py" --file-mode
```

### MCP Examples

Search all repositories:
```json
{
    "tool": "search_code",
    "args": {
        "query": "how to handle environment variables in tests",
        "num_results": 3
    }
}
```

Search in a specific repository:
```json
{
    "tool": "search_code",
    "args": {
        "query": "async def process_message(self, message: str):",
        "repository": "semantic-kernel/python",
        "num_results": 3
    }
}
```

Get a specific file:
```json
{
    "tool": "get_file",
    "args": {
        "file_path": "tests/utils.py",
        "repository": "semantic-kernel/python"
    }
}
```

Index a new repository:
```json
{
    "tool": "update_index",
    "args": {
        "repo_path": "/path/to/repository"
    }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)