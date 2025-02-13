import os
import pickle
import tempfile

import numpy as np
import pytest

from code_rag_server.vector_store import InMemoryVectorStore


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return {
        "file": "test.py",
        "repo": "test-repo",
        "code": "def test(): pass",
        "language": "python",
    }


@pytest.fixture
def temp_pickle_path():
    """Create a temporary file path for pickle testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Initialize with empty data structure
        pickle.dump(([], []), f)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_vector_store_initialization():
    """Test basic initialization of vector store."""
    store = InMemoryVectorStore()
    assert store.size == 0
    assert store.embeddings == []
    assert store.metadata == []


def test_add_vector(sample_embedding, sample_metadata):
    """Test adding a vector to the store."""
    store = InMemoryVectorStore()
    store.add(sample_embedding, sample_metadata)

    assert store.size == 1
    assert len(store.embeddings) == 1
    assert len(store.metadata) == 1
    assert store.metadata[0] == sample_metadata


def test_vector_normalization(sample_embedding, sample_metadata):
    """Test that vectors are normalized when added."""
    store = InMemoryVectorStore()
    store.add(sample_embedding, sample_metadata)

    # Check if the stored embedding is normalized
    stored_embedding = store.embeddings[0]
    norm = np.linalg.norm(stored_embedding)
    assert np.isclose(norm, 1.0)


def test_search(sample_embedding, sample_metadata):
    """Test vector search functionality."""
    store = InMemoryVectorStore()
    store.add(sample_embedding, sample_metadata)

    # Search with the same vector (should return perfect match)
    results = store.search(sample_embedding, top_k=1)
    assert len(results) == 1
    assert results[0][0] == sample_metadata  # Check metadata
    assert np.isclose(results[0][1], 1.0)  # Check similarity score


def test_search_empty_store(sample_embedding):
    """Test search on empty vector store."""
    store = InMemoryVectorStore()
    results = store.search(sample_embedding)
    assert len(results) == 0


def test_persistence(sample_embedding, sample_metadata, temp_pickle_path):
    """Test saving and loading the vector store."""
    # Create and save store
    store1 = InMemoryVectorStore(pickle_path=temp_pickle_path)
    store1.add(sample_embedding, sample_metadata)
    store1.save()

    # Load in new instance
    store2 = InMemoryVectorStore(pickle_path=temp_pickle_path)
    assert store2.size == 1
    assert len(store2.metadata) == 1
    assert store2.metadata[0] == sample_metadata
    np.testing.assert_array_almost_equal(store2.embeddings[0], store1.embeddings[0])


def test_clear(temp_pickle_path):
    """Test clearing the vector store."""
    # Initialize store with empty data
    store = InMemoryVectorStore(pickle_path=temp_pickle_path)

    # Add data
    store.add(np.array([0.1, 0.2, 0.3]), {"id": 1})
    assert store.size == 1

    # Clear store
    store.clear()
    assert store.size == 0
    assert len(store.embeddings) == 0
    assert len(store.metadata) == 0
    assert not os.path.exists(temp_pickle_path)


def test_multiple_vectors():
    """Test handling multiple vectors."""
    store = InMemoryVectorStore()

    # Add multiple vectors
    vectors = [
        (np.array([0.1, 0.2, 0.3]), {"id": 1}),
        (np.array([0.4, 0.5, 0.6]), {"id": 2}),
        (np.array([0.7, 0.8, 0.9]), {"id": 3}),
    ]

    for vec, meta in vectors:
        store.add(vec, meta)

    assert store.size == 3

    # Test search with top_k
    results = store.search(np.array([0.1, 0.2, 0.3]), top_k=2)
    assert len(results) == 2

    # First result should be the exact match
    assert results[0][0]["id"] == 1
    assert np.isclose(results[0][1], 1.0)


def test_save_load_empty_store(temp_pickle_path):
    """Test saving and loading an empty store."""
    store1 = InMemoryVectorStore(pickle_path=temp_pickle_path)
    store1.save()

    store2 = InMemoryVectorStore(pickle_path=temp_pickle_path)
    assert store2.size == 0
    assert len(store2.embeddings) == 0
    assert len(store2.metadata) == 0


def test_add_with_persistence(temp_pickle_path):
    """Test adding vectors with persistence enabled."""
    store = InMemoryVectorStore(pickle_path=temp_pickle_path)
    store.add(np.array([0.1, 0.2, 0.3]), {"id": 1})

    # Load new instance to verify persistence
    new_store = InMemoryVectorStore(pickle_path=temp_pickle_path)
    assert new_store.size == 1
    assert new_store.metadata[0]["id"] == 1


def test_multiple_saves(temp_pickle_path):
    """Test multiple save operations."""
    store = InMemoryVectorStore(pickle_path=temp_pickle_path)

    # Add and save multiple times
    for i in range(3):
        store.add(np.array([0.1 * i, 0.2 * i, 0.3 * i]), {"id": i})
        store.save()

    # Verify final state
    final_store = InMemoryVectorStore(pickle_path=temp_pickle_path)
    assert final_store.size == 3
    assert [m["id"] for m in final_store.metadata] == [0, 1, 2]
