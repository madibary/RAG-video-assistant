"""
store.py — Pinecone storage for YouTube transcript chunks.

Index:   configured via PINECONE_INDEX_NAME env var (default: "youtube-transcripts")
Metric:  cosine  (correct for L2-normalised BGE embeddings)
Vector ID format: "{video_id}_chunk_{chunk_index}"  (idempotent upserts)

Pinecone does not have a separate documents store, so the raw chunk text
is kept in the vector metadata under the key "text".  All retrieval results
therefore contain the passage text directly — no secondary lookup needed.
"""

from __future__ import annotations

import os
from typing import List

from pinecone import Pinecone, ServerlessSpec

from ingestion.chunker import Chunk

DEFAULT_INDEX_NAME = "youtube-transcripts"


# ---------------------------------------------------------------------------
# Client + index
# ---------------------------------------------------------------------------

def get_client() -> Pinecone:
    """
    Return a Pinecone client using PINECONE_API_KEY from the environment.
    Raises RuntimeError if the key is missing.
    """
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY environment variable is not set. "
            "Add it to your .env file or export it in your shell."
        )
    return Pinecone(api_key=api_key)


def get_or_create_index(
    client: Pinecone,
    index_name: str = DEFAULT_INDEX_NAME,
    dimension: int = 384,
    cloud: str = "aws",
    region: str = "us-east-1",
):
    """
    Get an existing Pinecone index or create it as a serverless index.

    `dimension` must match the embedding model's output size — pass
    embedder.get_dimension() from the caller so it's always in sync.
    The cloud/region defaults to AWS us-east-1 (Pinecone free tier).
    Override via PINECONE_CLOUD and PINECONE_REGION env vars.
    """
    cloud = os.environ.get("PINECONE_CLOUD", cloud)
    region = os.environ.get("PINECONE_REGION", region)

    existing = [idx.name for idx in client.list_indexes()]
    if index_name not in existing:
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    return client.Index(index_name)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    index,
    chunks: List[Chunk],
    embeddings: List[List[float]],
    video_id: str,
    video_title: str,
    video_url: str,
    batch_size: int = 100,
) -> None:
    """
    Upsert all chunks for a single video into Pinecone.

    Each vector is a dict with:
      id       — "{video_id}_chunk_{chunk_index}" (idempotent re-ingestion)
      values   — the embedding vector
      metadata — all searchable fields including the raw chunk text

    Metadata fields:
      text           — raw chunk text (returned in query results for reranking)
      video_id       — YouTube video ID
      video_title    — human-readable title
      video_url      — canonical watch URL
      timestamp_url  — deep-link to the exact moment (https://youtu.be/ID?t=N)
      start_time     — float seconds; supports $gte/$lte numeric filters
      end_time       — float seconds
      chunk_index    — 0-based position within the video
      total_chunks   — total chunks for this video
      source         — always "youtube"

    Vectors are sent in batches of `batch_size` (Pinecone recommends ≤100
    per request for optimal throughput).
    """
    if not chunks:
        return

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append(
            {
                "id": f"{video_id}_chunk_{chunk.chunk_index}",
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "video_id": video_id,
                    "video_title": video_title,
                    "video_url": video_url,
                    "timestamp_url": f"https://youtu.be/{video_id}?t={int(chunk.start_time)}",
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "source": "youtube",
                },
            }
        )

    # Send in batches
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])
