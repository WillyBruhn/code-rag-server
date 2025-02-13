from typing import List, Union

import httpx
import numpy as np
from rich.console import Console

console = Console()


class EmbeddingService:
    def __init__(self, api_url: str = "http://127.0.0.1:1235/v1/embeddings"):
        """Initialize the embedding service with the LLM Studio API endpoint."""
        self.api_url = api_url
        self.model = "text-embedding-nomic-embed-text-v1.5@q8_0"

    async def get_embedding(
        self, text: Union[str, List[str]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get embeddings for a single text or list of texts.

        Args:
            text: Single string or list of strings to embed

        Returns:
            Single embedding vector or list of embedding vectors

        Raises:
            ValueError: If input is empty
            HTTPError: If API call fails
        """
        try:
            # Validate input
            if isinstance(text, str) and not text.strip():
                raise ValueError("Empty input")
            elif isinstance(text, list):
                if not text or any(not t.strip() for t in text):
                    raise ValueError("Empty input")
                texts = text
            else:
                texts = [text]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json={"model": self.model, "input": texts},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                embeddings = [np.array(item["embedding"]) for item in data["data"]]
                return embeddings[0] if isinstance(text, str) else embeddings

        except httpx.HTTPError as e:
            console.print(f"[red]Error getting embeddings: {str(e)}[/red]")
            raise
        except Exception as e:
            console.print(f"[red]Unexpected error: {str(e)}[/red]")
            raise

    async def get_batch_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Get embeddings for a list of texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Maximum number of texts per batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        # Validate input
        if not texts:
            raise ValueError("Empty input")

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self.get_embedding(batch)
            all_embeddings.extend(
                batch_embeddings
                if isinstance(batch_embeddings, list)
                else [batch_embeddings]
            )

            # Log progress
            console.print(
                f"[green]Processed {min(i + batch_size, len(texts))}/{len(texts)} texts[/green]"
            )

        return all_embeddings
