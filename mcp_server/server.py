#!/usr/bin/env python3
"""
mcp_server/server.py — MCP server exposing YouTube transcript search.

Runs as a stdio MCP server that any MCP-compatible client can connect to.

Usage (standalone, for testing):
  python mcp_server/server.py

Usage (as a module):
  python -m mcp_server

Usage (registered in an MCP client config):
  {
    "command": "python",
    "args": ["/absolute/path/to/mcp_server/server.py"]
  }
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow imports from the project root (retrieval, ingestion, …)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from retrieval import search_transcripts as _search

load_dotenv()

mcp = FastMCP("youtube-rag")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seconds_to_hms(seconds: float) -> str:
    """Convert seconds to H:MM:SS string."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_transcripts(query: str, n_results: int = 5, video_id: str = "") -> str:
    """Search the YouTube transcript knowledge base for passages relevant to a query.

    Returns matching chunks with video titles and clickable timestamp URLs
    so the user can jump to that exact moment in the video.

    Args:
        query: The search query — phrase it as the key idea you are looking for.
        n_results: Number of chunks to return (default 5, max 20).
        video_id: Optional YouTube video ID to restrict search to one video.
                  Leave empty to search across all videos.
    """
    results = _search(
        query=query,
        n_results=min(n_results, 20),
        video_id=video_id or None,
    )
    if not results:
        return "No relevant content found for this query."

    for r in results:
        r["timestamp_hms"] = _seconds_to_hms(r["start_time"])
        r["citation"] = (
            f'[{r["video_title"]} – {r["timestamp_hms"]}]({r["timestamp_url"]})'
        )

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
