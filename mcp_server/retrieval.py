"""
retrieval.py — Pinecone query helpers for the RAG agent.

Used by mcp_server/server.py as the backend for the search_transcripts tool.
Can also be imported directly for custom retrieval logic.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from ingestion.embedder import embed_query
from ingestion.store import get_client

load_dotenv()

# ---------------------------------------------------------------------------
# CrossEncoder reranker (lazy singleton)
# ---------------------------------------------------------------------------

_RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANKER_MODEL)
    return _reranker


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_OVERSAMPLE_FACTOR = 4  # fetch this many × n_results candidates before reranking
_MAX_CANDIDATES = 100   # hard cap to keep Pinecone query fast


def search_transcripts(
    query: str,
    n_results: int = 5,
    video_id: Optional[str] = None,
) -> list[dict]:
    """Search Pinecone for relevant transcript chunks, then rerank with a CrossEncoder.

    Args:
        query:     Natural-language question or search phrase.
        n_results: How many chunks to return (default 5).
        video_id:  If provided, restrict results to a single video.

    Returns:
        List of dicts, each containing:
          text          — the chunk text
          video_title   — human-readable video title
          video_url     — canonical watch URL
          timestamp_url — deep-link to the exact moment (youtu.be/ID?t=N)
          start_time    — float seconds
          end_time      — float seconds
          video_id      — YouTube video ID
          score         — cosine similarity score
          rerank_score  — CrossEncoder relevance score
    """
    pc = get_client()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "youtube-transcripts")
    index = pc.Index(index_name)

    query_embedding = embed_query(query)
    filter_ = {"video_id": {"$eq": video_id}} if video_id else None

    # Oversample from Pinecone so the reranker has more candidates to work with
    top_k = min(n_results * _OVERSAMPLE_FACTOR, _MAX_CANDIDATES)
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_,
    )

    candidates = [
        {
            "text": match.metadata.get("text", ""),
            "video_title": match.metadata.get("video_title", ""),
            "video_url": match.metadata.get("video_url", ""),
            "timestamp_url": match.metadata.get("timestamp_url", ""),
            "start_time": match.metadata.get("start_time", 0.0),
            "end_time": match.metadata.get("end_time", 0.0),
            "video_id": match.metadata.get("video_id", ""),
            "score": round(float(match.score), 4),
        }
        for match in response.matches
    ]

    if not candidates:
        return []

    # Rerank with CrossEncoder: score each (query, passage) pair
    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs).tolist()

    for candidate, rerank_score in zip(candidates, rerank_scores):
        candidate["rerank_score"] = round(float(rerank_score), 4)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    return candidates[:n_results]
