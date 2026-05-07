"""
transcript.py — YouTube transcript and metadata fetching.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import List

import yt_dlp
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,  # kept for broad exception matching
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RawSegment:
    text: str
    start: float    # seconds from video start
    duration: float # length of this segment in seconds


@dataclass
class VideoData:
    video_id: str
    title: str
    url: str                    # canonical https://www.youtube.com/watch?v=...
    segments: List[RawSegment]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class TranscriptUnavailableError(Exception):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """
    Parse any common YouTube URL form and return the bare video ID.

    Handles:
      https://www.youtube.com/watch?v=XXXXXXXXXXX
      https://youtu.be/XXXXXXXXXXX
      https://www.youtube.com/embed/XXXXXXXXXXX
      https://www.youtube.com/shorts/XXXXXXXXXXX

    Raises ValueError if no ID can be parsed.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().lstrip("www.")

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    if host in ("youtube.com", "m.youtube.com"):
        # /watch?v=...
        qs = urllib.parse.parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

        # /embed/<id> or /shorts/<id>
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("embed", "shorts", "v", "e"):
            return parts[1]

    raise ValueError(f"Could not extract a YouTube video ID from URL: {url!r}")


def fetch_metadata(video_id: str) -> dict:
    """
    Use yt-dlp in no-download mode to retrieve video metadata.
    Returns {"title": str}.  Falls back to "Unknown Title" on any error.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=False,
            )
            return {"title": info.get("title") or "Unknown Title"}
    except Exception:
        return {"title": "Unknown Title"}


def _to_raw_segment(seg) -> RawSegment:
    """
    Normalise a transcript snippet to a RawSegment.

    Handles both the old dict-style API (< 0.6.0) and the new
    attribute-style FetchedTranscriptSnippet objects (>= 0.6.0).
    """
    if isinstance(seg, dict):
        return RawSegment(
            text=seg["text"],
            start=float(seg["start"]),
            duration=float(seg["duration"]),
        )
    return RawSegment(
        text=seg.text,
        start=float(seg.start),
        duration=float(seg.duration),
    )


def fetch_transcript(video_id: str) -> List[RawSegment]:
    """
    Fetch the transcript for a video as a list of RawSegments.

    Strategy:
      1. Try English first via the instance API (>= 0.6.0).
      2. Fall back to the first available language.

    Raises TranscriptUnavailableError if nothing is accessible.
    """
    api = YouTubeTranscriptApi()

    try:
        raw = api.fetch(video_id, languages=("en",))
    except (NoTranscriptFound, TranscriptsDisabled):
        try:
            # No English transcript — fetch whatever language is available
            raw = api.fetch(video_id)
        except Exception as exc:
            raise TranscriptUnavailableError(
                f"No transcript available for video {video_id!r}: {exc}"
            ) from exc
    except Exception as exc:
        raise TranscriptUnavailableError(
            f"Failed to fetch transcript for video {video_id!r}: {exc}"
        ) from exc

    return [_to_raw_segment(seg) for seg in raw]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_video(url: str) -> VideoData:
    """
    Parse a YouTube URL, fetch transcript and metadata, and return a VideoData.
    This is the only function ingest.py needs to call from this module.
    """
    video_id = extract_video_id(url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    metadata = fetch_metadata(video_id)
    segments = fetch_transcript(video_id)

    return VideoData(
        video_id=video_id,
        title=metadata["title"],
        url=canonical_url,
        segments=segments,
    )
