from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import HTTPError

from code_rag_server.embeddings import EmbeddingService


@pytest.fixture
def embedding_service():
    """Create an embedding service instance."""
    return EmbeddingService()


@pytest.fixture
def mock_response():
    """Create a mock successful response."""
    return {
        "data": [
            {
                "embedding": [0.1] * 768,  # Match actual embedding dimension
                "index": 0,
            }
        ],
        "model": "text-embedding-nomic-embed-text-v1.5@q8_0",
    }


@pytest.fixture
def mock_batch_response():
    """Create a mock response for batch processing."""
    return {
        "data": [
            {"embedding": [0.1] * 768, "index": 0},
            {"embedding": [0.2] * 768, "index": 1},
            {"embedding": [0.3] * 768, "index": 2},
        ],
        "model": "text-embedding-nomic-embed-text-v1.5@q8_0",
    }


@pytest.mark.asyncio
async def test_get_single_embedding(embedding_service, mock_response):
    """Test getting embedding for a single text."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(), json=MagicMock(return_value=mock_response)
        )

        embedding = await embedding_service.get_embedding("test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)  # Check correct dimension
        assert np.isclose(embedding[0], 0.1)


@pytest.mark.asyncio
async def test_batch_embeddings(embedding_service, mock_batch_response):
    """Test getting embeddings for multiple texts."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value=mock_batch_response),
        )

        texts = ["text1", "text2", "text3"]
        embeddings = await embedding_service.get_embedding(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (768,) for emb in embeddings)


@pytest.mark.asyncio
async def test_batch_processing(embedding_service, mock_batch_response):
    """Test batch processing with specific batch size."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value=mock_batch_response),
        )

        texts = ["text1", "text2", "text3", "text4", "text5"]
        embeddings = await embedding_service.get_batch_embeddings(texts, batch_size=2)

        assert isinstance(embeddings, list)
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (768,) for emb in embeddings)


@pytest.mark.asyncio
async def test_http_error_handling(embedding_service):
    """Test handling of HTTP errors."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = HTTPError("API Error")

        with pytest.raises(HTTPError):
            await embedding_service.get_embedding("test text")


@pytest.mark.asyncio
async def test_invalid_response(embedding_service):
    """Test handling of invalid API responses."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"invalid": "response"}),
        )

        with pytest.raises(KeyError):
            await embedding_service.get_embedding("test text")


@pytest.mark.asyncio
async def test_custom_api_url():
    """Test using a custom API URL."""
    custom_url = "http://custom-api.local/embeddings"
    service = EmbeddingService(api_url=custom_url)

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={"data": [{"embedding": [0.1] * 768, "index": 0}]}
            ),
        )

        await service.get_embedding("test")

        # Verify the custom URL was used
        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == custom_url


@pytest.mark.asyncio
async def test_empty_input_handling(embedding_service):
    """Test handling of empty input."""
    # Test empty string
    with pytest.raises(ValueError, match="Empty input"):
        await embedding_service.get_embedding("")

    # Test empty list
    with pytest.raises(ValueError, match="Empty input"):
        await embedding_service.get_batch_embeddings([])

    # Test list with empty string
    with pytest.raises(ValueError, match="Empty input"):
        await embedding_service.get_embedding([""])


@pytest.mark.asyncio
async def test_large_batch_processing(embedding_service):
    """Test processing of large batches."""
    batch_size = 32
    total_texts = 100
    texts = ["text"] * total_texts

    async def mock_post(*args, **kwargs):
        # Create response based on the input size
        input_texts = kwargs["json"]["input"]
        return MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={
                    "data": [
                        {"embedding": [0.1] * 768, "index": i}
                        for i in range(len(input_texts))
                    ]
                }
            ),
        )

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        embeddings = await embedding_service.get_batch_embeddings(
            texts, batch_size=batch_size
        )

        # Verify results
        assert len(embeddings) == total_texts
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (768,) for emb in embeddings)
