import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm


class InMemoryVectorStore:
    def __init__(self, pickle_path: Optional[str] = None):
        """Initialize an in-memory vector store with optional persistence."""
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.pickle_path = pickle_path

        if pickle_path and os.path.exists(pickle_path):
            self.load()

    def add(self, embedding: np.ndarray, metadata: Dict) -> None:
        """Add an embedding vector and its metadata to the store."""
        # Normalize the embedding vector
        normalized_embedding = embedding / norm(embedding)
        self.embeddings.append(normalized_embedding)
        self.metadata.append(metadata)

        # Auto-save if pickle path is set
        if self.pickle_path:
            self.save()

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """Search for most similar vectors using cosine similarity."""
        if not self.embeddings:
            return []

        # Normalize query vector
        query_embedding = query_embedding / norm(query_embedding)

        # Convert list to numpy array for faster computation
        embeddings_array = np.array(self.embeddings)

        # Compute cosine similarities
        similarities = np.dot(embeddings_array, query_embedding)

        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return metadata and scores
        return [(self.metadata[i], float(similarities[i])) for i in top_indices]

    def save(self) -> None:
        """Save the vector store to disk."""
        if self.pickle_path:
            with open(self.pickle_path, "wb") as f:
                pickle.dump((self.embeddings, self.metadata), f)

    def load(self) -> None:
        """Load the vector store from disk."""
        if self.pickle_path and os.path.exists(self.pickle_path):
            with open(self.pickle_path, "rb") as f:
                self.embeddings, self.metadata = pickle.load(f)

    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        self.embeddings = []
        self.metadata = []
        if self.pickle_path and os.path.exists(self.pickle_path):
            os.remove(self.pickle_path)

    @property
    def size(self) -> int:
        """Return the number of vectors in the store."""
        return len(self.embeddings)
