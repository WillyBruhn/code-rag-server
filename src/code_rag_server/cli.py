import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from .embeddings import EmbeddingService
from .vector_store import InMemoryVectorStore

console = Console()


@click.group()
def cli():
    """Code RAG Server CLI - Search and index code repositories."""
    pass


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
def index(repo_path: Path):
    """Index a repository for searching."""
    try:
        repo_path = Path(repo_path).resolve()
        console.print(f"Starting indexing process for {repo_path}")
        asyncio.run(_index_repository(repo_path))
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(2)


async def _index_repository(repo_path: Path):
    """Index a repository asynchronously."""
    from .server import update_index

    await update_index(repo_path)


@cli.command()
@click.argument("query")
@click.option("-n", "--num-results", default=5, help="Number of results to return")
@click.option(
    "--file-mode/--no-file-mode", default=False, help="Search for files by path"
)
@click.option("--repository", help="Specific repository to search in")
def search(query: str, num_results: int, file_mode: bool, repository: str = None):
    """Search through indexed code repositories."""
    try:
        if num_results < 1:
            raise click.BadParameter("Number of results must be positive")
        asyncio.run(_search_code(query, num_results, file_mode, repository))
    except click.BadParameter as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(2)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(2)


async def _search_code(
    query: str, num_results: int, file_mode: bool, repository: str = None
):
    """Search code asynchronously."""
    indices_dir = Path("indices")
    if not indices_dir.exists() or not any(indices_dir.glob("**/*.pkl")):
        console.print("[red]No indexes found. Please index a repository first.[/red]")
        sys.exit(2)

    # Get list of available repositories
    indices = list(indices_dir.glob("**/*.pkl"))
    if repository:
        repo_path = indices_dir / f"{repository}.pkl"
        if not repo_path.exists():
            console.print(f"[red]Repository {repository} not found in indices.[/red]")
            sys.exit(2)
        indices = [repo_path]

    console.print(f"Found {len(indices)} indexed repositories.")
    embedding_service = EmbeddingService()

    for idx_path in indices:
        repo_name = idx_path.stem
        console.print(f"Searching in {repo_name}...")

        store = InMemoryVectorStore(pickle_path=str(idx_path))

        if file_mode:
            # File path search
            matches = []
            for metadata in store.metadata:
                if query.lower() in metadata["file"].lower():
                    matches.append(metadata)

            if not matches:
                console.print("No matching files found.")
                continue

            console.print("Search Results:\n")
            for match in matches[:num_results]:
                console.print(f"File: {match['file']}")
                console.print(f"Repository: {match['repo']}")
                console.print(
                    f"Code:\n```{match.get('language', '')}\n{match['code']}\n```\n"
                )

        else:
            # Semantic search
            query_embedding = await embedding_service.get_embedding(query)
            results = store.search(query_embedding, top_k=num_results)

            if not results:
                console.print("No matches found.")
                return

            console.print("Search Results:\n")
            for metadata, score in results:
                console.print(f"File: {metadata['file']}")
                console.print(f"Repository: {metadata['repo']}")
                console.print(f"Score: {score:.2f}")
                console.print(
                    f"Code:\n```{metadata.get('language', '')}\n{metadata['code']}\n```\n"
                )
