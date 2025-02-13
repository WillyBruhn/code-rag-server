import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from rich.console import Console
from tqdm import tqdm

from .embeddings import EmbeddingService
from .vector_store import InMemoryVectorStore

# Initialize components
console = Console()
server = Server("code-rag-server")
embedding_service = EmbeddingService()

# Setup vector store with persistence
indices_dir = Path("indices")
indices_dir.mkdir(parents=True, exist_ok=True)
vector_store: Optional[InMemoryVectorStore] = None


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="clone_repo",
            description="Clone a GitHub repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "repository_url": {
                        "type": "string",
                        "description": "GitHub repository URL (e.g., https://github.com/owner/repo)",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Optional: specific branch to clone",
                        "required": False,
                    },
                },
                "required": ["repository_url"],
            },
        ),
        types.Tool(
            name="search_code",
            description="Search for code using natural language or code snippets",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (can be natural language or code)",
                    },
                    "repository": {
                        "type": "string",
                        "description": "Optional: Specific repository to search in",
                        "required": False,
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                        "minimum": 1,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_file",
            description="Get the content of a specific file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path relative to repository",
                    },
                    "repository": {"type": "string", "description": "Repository name"},
                },
                "required": ["file_path", "repository"],
            },
        ),
        types.Tool(
            name="update_index",
            description="Update the code index for a repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to repository directory",
                    }
                },
                "required": ["repo_path"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """Handle tool execution requests."""
    if not arguments:
        raise ValueError("Missing arguments")

    if name == "clone_repo":
        repository_url = arguments["repository_url"]
        branch = arguments.get("branch")

        try:
            repo_path = _clone_github_repo(repository_url, branch)
            relative_path = repo_path.relative_to(Path("github_repos"))

            return [
                types.TextContent(
                    type="text",
                    text=f"Successfully cloned repository to github_repos/{relative_path}\n"
                    f"To index this repository, use the update_index tool with:\n"
                    f"repo_path: {repo_path}",
                )
            ]

        except (ValueError, subprocess.CalledProcessError) as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "search_code":
        query = arguments["query"]
        num_results = arguments.get("num_results", 5)
        repository = arguments.get("repository")

        if repository:
            pickle_file = indices_dir / f"{repository}.pkl"
            if not pickle_file.exists():
                return [
                    types.TextContent(
                        type="text",
                        text=f"Repository {repository} is not indexed. Please update the index first.",
                    )
                ]
            pickle_files = [pickle_file]
        else:
            pickle_files = list(indices_dir.glob("**/*.pkl"))

        if not pickle_files:
            return [
                types.TextContent(
                    type="text", text="No indexes found. Please update the index first."
                )
            ]

        # Load vector stores and search
        all_results = []
        for pickle_file in pickle_files:
            store = InMemoryVectorStore(pickle_path=str(pickle_file))
            if store.size > 0:
                query_embedding = await embedding_service.get_embedding(query)
                results = store.search(query_embedding, top_k=num_results)
                all_results.extend(results)

        if not all_results:
            return [types.TextContent(type="text", text="No matches found.")]

        # Sort results by score and format output
        all_results.sort(key=lambda x: x[1], reverse=True)
        all_results = all_results[:num_results]

        formatted_results = []
        for metadata, score in all_results:
            formatted_results.append(
                f"File: {metadata['file']}\n"
                f"Repository: {metadata['repo']}\n"
                f"Score: {score:.2f}\n"
                f"Code:\n```{metadata.get('language', '')}\n{metadata['code']}\n```\n"
            )

        return [types.TextContent(type="text", text="\n".join(formatted_results))]

    elif name == "get_file":
        file_path = arguments["file_path"]
        repository = arguments["repository"]
        pickle_file = indices_dir / f"{repository}.pkl"

        if not pickle_file.exists():
            return [
                types.TextContent(
                    type="text",
                    text=f"Repository {repository} is not indexed. Please update the index first.",
                )
            ]

        store = InMemoryVectorStore(pickle_path=str(pickle_file))
        for metadata in store.metadata:
            if metadata["file"] == file_path:
                return [
                    types.TextContent(
                        type="text",
                        text=f"File: {metadata['file']}\n"
                        f"Repository: {metadata['repo']}\n"
                        f"Code:\n```{metadata.get('language', '')}\n{metadata['code']}\n```",
                    )
                ]

        return [
            types.TextContent(
                type="text",
                text=f"File {file_path} not found in repository {repository}.",
            )
        ]

    elif name == "update_index":
        repo_path = Path(arguments["repo_path"])
        return await update_index(repo_path)

    raise ValueError(f"Unknown tool: {name}")


async def update_index(repo_path: Path) -> List[types.TextContent]:
    """Update the code index for a repository."""
    if not repo_path.exists():
        return [
            types.TextContent(
                type="text", text=f"Repository path does not exist: {repo_path}"
            )
        ]

    relative_path = repo_path.relative_to(Path("github_repos"))
    pickle_file = indices_dir / f"{relative_path}.pkl"
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    store = InMemoryVectorStore(pickle_path=str(pickle_file))

    # Process files
    indexed_files = 0
    file_contents = []
    file_metadata = []

    console.print("[green]Starting indexing process...[/green]")
    for file_path in tqdm(list(repo_path.rglob("*")), desc="Scanning files"):
        if file_path.is_file() and not _should_ignore(file_path):
            try:
                content = file_path.read_text()
                file_contents.append(content)
                file_metadata.append(
                    {
                        "file": str(file_path.relative_to(repo_path)),
                        "repo": str(relative_path),
                        "code": content,
                        "language": _guess_language(file_path),
                    }
                )
                indexed_files += 1
            except Exception as e:
                console.print(f"[red]Error processing {file_path}: {str(e)}[/red]")

    if file_contents:
        console.print(
            f"[green]Generating embeddings for {len(file_contents)} files...[/green]"
        )
        embeddings = await embedding_service.get_batch_embeddings(file_contents)

        console.print("[green]Storing embeddings...[/green]")
        for embedding, metadata in tqdm(
            zip(embeddings, file_metadata, strict=False), desc="Storing embeddings"
        ):
            store.add(embedding, metadata)

        return [
            types.TextContent(
                type="text",
                text=f"Successfully indexed {indexed_files} files from {relative_path}",
            )
        ]
    else:
        return [
            types.TextContent(type="text", text=f"No files to index in {relative_path}")
        ]


def _should_ignore(path: Path) -> bool:
    """Check if a file should be ignored."""
    ignore_patterns = {
        ".git",
        "__pycache__",
        "node_modules",
        "venv",
        ".env",
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dll",
        ".dylib",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".pdf",
        ".zip",
    }
    return any(part in ignore_patterns for part in path.parts)


def _validate_github_url(url: str) -> Tuple[str, str]:
    """Validate GitHub URL and extract owner/repo."""
    https_pattern = r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$"
    ssh_pattern = r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$"

    https_match = re.match(https_pattern, url)
    ssh_match = re.match(ssh_pattern, url)

    if https_match:
        return https_match.groups()
    elif ssh_match:
        return ssh_match.groups()
    else:
        raise ValueError(
            "Invalid GitHub URL. Must be in format: "
            "https://github.com/owner/repo or "
            "git@github.com:owner/repo"
        )


def _clone_github_repo(url: str, branch: Optional[str] = None) -> Path:
    """Clone a GitHub repository."""
    owner, repo = _validate_github_url(url)

    base_path = Path("github_repos")
    repo_path = base_path / owner / repo

    if repo_path.exists():
        raise ValueError(f"Repository already exists at {repo_path}")

    repo_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone"]
    if branch:
        cmd.extend(["-b", branch])
    cmd.extend([url, str(repo_path)])

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    return repo_path


def _guess_language(path: Path) -> str:
    """Guess programming language from file extension."""
    ext_to_lang = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
    }
    return ext_to_lang.get(path.suffix.lower(), "")


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="code-rag-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
