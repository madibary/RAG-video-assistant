"""
chunker.py — Semantic chunking of YouTube transcript segments.

Algorithm:
  1. Concatenate all segments into one text string, tracking character offsets
     so every sentence can be mapped back to source timestamps.
  2. Split the full text into sentences with spaCy's rule-based sentencizer.
     (No model download required — ships with spaCy core.)
  3. Embed every sentence with the passage model (BAAI/bge-small-en-v1.5).
  4. Compute cosine similarity between consecutive sentence embeddings.
     (dot product is sufficient because embeddings are L2-normalised.)
  5. Mark a breakpoint wherever similarity falls below the Nth percentile
     (default: 10th).  These are topic-shift boundaries.
  6. Group sentences between breakpoints into Chunk objects, preserving
     the start/end timestamps from the original segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import spacy
import numpy as np

from ingestion.transcript import RawSegment
from ingestion import embedder as _embedder

# Module-level spaCy pipeline singleton (lazy-loaded on first use).
# Uses spacy.blank("en") + the rule-based sentencizer — no model download needed.
_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _nlp = nlp
    return _nlp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SentenceRecord:
    text: str
    start_time: float   # seconds — first segment that contains this sentence
    end_time: float     # seconds — last segment that contains this sentence


@dataclass
class Chunk:
    text: str
    start_time: float
    end_time: float
    chunk_index: int
    video_id: str
    total_chunks: int = field(default=0, compare=False)  # backfilled after build


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_offset_index(
    segments: List[RawSegment],
) -> tuple[str, list[tuple[int, int, RawSegment]]]:
    """
    Concatenate segment texts (separated by a single space) and build an index
    mapping character spans → RawSegment.

    Returns:
        full_text: the concatenated string
        offset_index: sorted list of (char_start, char_end, segment) tuples
    """
    parts: list[str] = []
    offset_index: list[tuple[int, int, RawSegment]] = []
    cursor = 0

    for seg in segments:
        text = seg.text.strip()
        char_start = cursor
        char_end = cursor + len(text)
        offset_index.append((char_start, char_end, seg))
        parts.append(text)
        cursor = char_end + 1  # +1 for the space separator

    full_text = " ".join(parts)
    return full_text, offset_index


def _sentence_timestamps(
    sentence: str,
    sent_char_start: int,
    sent_char_end: int,
    offset_index: list[tuple[int, int, RawSegment]],
) -> tuple[float, float]:
    """
    Given a sentence's character span in the full text, find all segments that
    overlap with it and return (min_start, max_end) in seconds.

    Uses binary search on offset_index (sorted by char_start) to find the first
    potentially overlapping segment, then scans forward while overlapping.
    """
    # Binary search: find the last segment whose char_start <= sent_char_start
    lo, hi = 0, len(offset_index) - 1
    idx = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if offset_index[mid][0] <= sent_char_start:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Walk forward collecting all overlapping segments
    min_start = float("inf")
    max_end = float("-inf")

    i = idx
    while i < len(offset_index):
        seg_char_start, seg_char_end, seg = offset_index[i]
        # Stop once segment starts beyond the sentence end
        if seg_char_start > sent_char_end:
            break
        # Overlap condition: not (seg_end < sent_start or seg_start > sent_end)
        if seg_char_end >= sent_char_start and seg_char_start <= sent_char_end:
            min_start = min(min_start, seg.start)
            max_end = max(max_end, seg.start + seg.duration)
        i += 1

    # Fallback (shouldn't happen with valid input)
    if min_start == float("inf"):
        seg = offset_index[idx][2]
        min_start = seg.start
        max_end = seg.start + seg.duration

    return min_start, max_end


def _sentences_with_timestamps(
    full_text: str,
    offset_index: list[tuple[int, int, RawSegment]],
) -> List[SentenceRecord]:
    """
    Tokenize full_text into sentences with spaCy and resolve timestamps for each.

    spaCy's sentencizer provides exact character spans (start_char, end_char)
    directly — no string searching needed.
    """
    nlp = _get_nlp()
    doc = nlp(full_text)
    records: List[SentenceRecord] = []

    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        char_start = sent.start_char
        char_end = sent.end_char - 1  # end_char is exclusive in spaCy

        start_time, end_time = _sentence_timestamps(
            text, char_start, char_end, offset_index
        )
        records.append(SentenceRecord(text=text, start_time=start_time, end_time=end_time))

    return records


def _find_breakpoints(
    embeddings: np.ndarray,
    percentile: int,
) -> list[int]:
    """
    Return indices i where a new chunk should start AFTER sentence i.

    Computes cosine similarity between consecutive sentence embeddings
    (dot product since embeddings are normalised), then marks breakpoints
    wherever the similarity falls below the given percentile threshold.
    """
    if len(embeddings) < 2:
        return []

    # dot product of consecutive pairs → shape (N-1,)
    similarities = np.einsum("ij,ij->i", embeddings[:-1], embeddings[1:])

    threshold = float(np.percentile(similarities, percentile))
    breakpoints = [int(i) for i in np.where(similarities < threshold)[0]]
    return breakpoints


def _assemble_chunks(
    sentence_records: List[SentenceRecord],
    breakpoints: list[int],
    video_id: str,
) -> List[Chunk]:
    """
    Group sentences between breakpoints into Chunk objects.
    `breakpoints` contains the indices of sentences after which a new chunk starts.
    """
    breakpoint_set = set(breakpoints)
    chunks: List[Chunk] = []

    group: List[SentenceRecord] = []
    for i, record in enumerate(sentence_records):
        group.append(record)
        if i in breakpoint_set or i == len(sentence_records) - 1:
            chunk_text = " ".join(r.text for r in group)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_time=group[0].start_time,
                    end_time=group[-1].end_time,
                    chunk_index=len(chunks),
                    video_id=video_id,
                )
            )
            group = []

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_segments(
    segments: List[RawSegment],
    video_id: str,
    breakpoint_percentile: int = 10,
    model_name: str = "BAAI/bge-small-en-v1.5",
    show_progress: bool = False,
) -> List[Chunk]:
    """
    Semantically chunk a list of transcript segments.

    Steps:
      1. Build full text + character offset index from segments.
      2. Tokenize into sentences; resolve timestamps for each sentence.
      3. Embed all sentences.
      4. Find topic-shift breakpoints via consecutive cosine similarity.
      5. Assemble sentences into Chunk objects.
      6. Backfill total_chunks on every chunk.

    Args:
        segments:              Raw transcript segments from transcript.py.
        video_id:              YouTube video ID (stored in each Chunk).
        breakpoint_percentile: Similarities below this percentile trigger a
                               chunk boundary (default 10 → bottom 10%).
        model_name:            Sentence-transformers model to use.
        show_progress:         Show a tqdm bar during sentence embedding.

    Returns:
        List of Chunk objects, ordered by position in the video.
    """
    if not segments:
        return []

    # Step 1
    full_text, offset_index = _build_offset_index(segments)

    # Step 2
    sentence_records = _sentences_with_timestamps(full_text, offset_index)

    if not sentence_records:
        return []

    # Step 3
    sentence_texts = [r.text for r in sentence_records]
    embeddings = _embedder.embed_sentences(
        sentence_texts,
        model_name=model_name,
        show_progress=show_progress,
    )

    # Step 4
    breakpoints = _find_breakpoints(embeddings, breakpoint_percentile)

    # Step 5
    chunks = _assemble_chunks(sentence_records, breakpoints, video_id)

    # Step 6 — backfill total_chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total

    return chunks
