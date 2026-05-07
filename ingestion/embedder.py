"""
embedder.py — Sentence-transformers embedding wrapper.

Uses BAAI/bge-small-en-v1.5 by default.  The model is loaded once per process
and reused across all calls (lazy singleton).

BGE models use asymmetric prompts:
  documents → prompt_name="document"
  queries   → prompt_name="query"
Passing the correct prompt_name is important for retrieval quality.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_model: SentenceTransformer | None = None


def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load the model once per process and cache it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def embed_sentences(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of sentence strings and return a numpy array of shape (N, D).

    Used internally by chunker.py for fast dot-product similarity computation.
    normalize_embeddings=True ensures dot product == cosine similarity.
    """
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        prompt_name="document",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings  # shape (N, D)


def embed_chunks(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = False,
) -> List[List[float]]:
    """
    Embed a list of chunk texts for storage in Pinecone.
    Returns Python lists (Pinecone requires List[float], not numpy arrays).
    """
    embeddings = embed_sentences(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    return [emb.tolist() for emb in embeddings]


def get_dimension(model_name: str = DEFAULT_MODEL) -> int:
    """Return the embedding dimension of the loaded model."""
    return _get_model(model_name).get_sentence_embedding_dimension()


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL,
) -> List[float]:
    """
    Embed a single query string for retrieval at query time.
    Uses prompt_name="query" (BGE asymmetric convention).
    """
    model = _get_model(model_name)
    embedding = model.encode(
        query,
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()
