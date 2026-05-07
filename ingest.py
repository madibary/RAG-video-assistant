#!/usr/bin/env python3
"""
ingest.py — YouTube RAG ingestion pipeline CLI.

Usage:
  python ingest.py <url1> [url2 ...] [options]
  python ingest.py --urls-file urls.txt [options]

Examples:
  python ingest.py "https://youtu.be/dQw4w9WgXcQ"
  python ingest.py "https://youtu.be/abc" "https://youtu.be/def"
  python ingest.py --urls-file videos.txt --breakpoint-percentile 15 --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from ingestion.transcript import load_video, TranscriptUnavailableError
from ingestion.chunker import chunk_segments
from ingestion.embedder import embed_chunks, get_dimension
from ingestion.store import get_client, get_or_create_index, upsert_chunks


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Ingest YouTube videos into a Pinecone RAG vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "urls",
        nargs="*",
        metavar="URL",
        help="One or more YouTube URLs to ingest.",
    )
    parser.add_argument(
        "--urls-file",
        metavar="FILE",
        help="Path to a text file with one YouTube URL per line "
             "(blank lines and lines starting with # are ignored).",
    )
    parser.add_argument(
        "--index-name",
        metavar="NAME",
        default=None,
        help="Pinecone index name. "
             "Defaults to env var PINECONE_INDEX_NAME or 'youtube-transcripts'.",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="Sentence-transformers model for embedding. "
             "Defaults to env var EMBEDDING_MODEL or BAAI/bge-small-en-v1.5.",
    )
    parser.add_argument(
        "--breakpoint-percentile",
        type=int,
        default=10,
        metavar="N",
        help="Percentile threshold for semantic chunk breakpoints (1-99). "
             "Lower values → fewer, larger chunks. Higher → more, smaller chunks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-chunk details during ingestion.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# URL loading
# ---------------------------------------------------------------------------

def load_urls_from_file(path: str) -> list[str]:
    """Read URLs from a text file, skipping blank lines and # comments."""
    urls = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def collect_urls(args: argparse.Namespace) -> list[str]:
    """Merge positional URLs and --urls-file, deduplicate preserving order."""
    urls: list[str] = list(args.urls)

    if args.urls_file:
        urls.extend(load_urls_from_file(args.urls_file))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    return unique


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------

def process_video(
    url: str,
    index,
    model_name: str,
    breakpoint_percentile: int,
    verbose: bool,
) -> dict:
    """
    Run the full pipeline for a single video.

    Returns a summary dict:
      {url, video_id, title, chunks, status, error}
    """
    result = {"url": url, "video_id": None, "title": None, "chunks": 0,
              "status": "error", "error": None}
    try:
        # 1. Fetch transcript + metadata
        video = load_video(url)
        result["video_id"] = video.video_id
        result["title"] = video.title

        if verbose:
            print(f"  [{video.video_id}] \"{video.title}\" — "
                  f"{len(video.segments)} segments")

        # 2. Semantic chunking (also embeds sentences internally)
        chunks = chunk_segments(
            video.segments,
            video_id=video.video_id,
            breakpoint_percentile=breakpoint_percentile,
            model_name=model_name,
            show_progress=verbose,
        )

        if not chunks:
            result["error"] = "No chunks produced (transcript may be empty)."
            return result

        result["chunks"] = len(chunks)

        if verbose:
            for c in chunks:
                print(f"    chunk {c.chunk_index:3d} | "
                      f"{c.start_time:7.1f}s – {c.end_time:7.1f}s | "
                      f"{len(c.text.split()):4d} words")

        # 3. Embed chunks for storage
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_chunks(chunk_texts, model_name=model_name)

        # 4. Store in Pinecone
        upsert_chunks(
            index=index,
            chunks=chunks,
            embeddings=embeddings,
            video_id=video.video_id,
            video_title=video.title,
            video_url=video.url,
        )

        result["status"] = "ok"
        return result

    except TranscriptUnavailableError as exc:
        result["error"] = str(exc)
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args = parse_args()

    # Resolve config from args → env vars → defaults
    index_name = (
        args.index_name
        or os.environ.get("PINECONE_INDEX_NAME", "youtube-transcripts")
    )
    model_name = (
        args.model
        or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # Collect and validate URLs
    urls = collect_urls(args)
    if not urls:
        print("Error: no URLs provided. Pass URLs as arguments or use --urls-file.",
              file=sys.stderr)
        sys.exit(1)

    # Initialise Pinecone (fatal if this fails)
    try:
        client = get_client()
        index = get_or_create_index(client, index_name=index_name, dimension=get_dimension(model_name))
        stats = index.describe_index_stats()
        print(f"Pinecone ready (index: '{index_name}', "
              f"{stats.total_vector_count} existing vectors)")
    except Exception as exc:
        print(f"Fatal: could not initialise Pinecone — {exc}", file=sys.stderr)
        sys.exit(1)

    # Process each video
    results = []
    iterator = tqdm(urls, desc="Ingesting videos", unit="video") if len(urls) > 1 else urls

    for url in iterator:
        if len(urls) == 1:
            print(f"Processing: {url}")

        summary = process_video(
            url=url,
            index=index,
            model_name=model_name,
            breakpoint_percentile=args.breakpoint_percentile,
            verbose=args.verbose,
        )
        results.append(summary)

        status_icon = "✓" if summary["status"] == "ok" else "✗"
        title = summary["title"] or "(unknown)"
        if summary["status"] == "ok":
            print(f"  {status_icon} {title!r} — {summary['chunks']} chunks stored")
        else:
            print(f"  {status_icon} {url} — FAILED: {summary['error']}", file=sys.stderr)

    # Final summary
    succeeded = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - succeeded
    print(f"\nDone: {succeeded} succeeded, {failed} failed.")
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
