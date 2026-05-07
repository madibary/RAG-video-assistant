# YouTube RAG

A retrieval-augmented generation pipeline that ingests YouTube video transcripts, indexes them in a vector database, and lets you ask natural-language questions about the content — with answers cited to exact video timestamps and followed by related academic paper suggestions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Ingestion pipeline                   │
│                                                          │
│  YouTube URL → transcript (youtube-transcript-api)       │
│             → semantic chunks with timestamps            │
│             → embeddings (BAAI/bge-small-en-v1.5)        │
│             → Pinecone vector store                      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                      MCP servers                         │
│                                                          │
│  mcp_server/server.py   → search_transcripts tool        │
│    └─ retrieval.py        (Pinecone + CrossEncoder)       │
│                                                          │
│  mcp-simple-arxiv       → arXiv paper search tool        │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Agent (LangGraph)                     │
│                                                          │
│  MultiServerMCPClient → tools from both servers         │
│  StateGraph (ReAct)   → LLM decides when to call tools  │
│  ChatGroq             → qwen/qwen3-32b (free tier)       │
│                                                          │
│  Output: answer with [Video – H:MM:SS] citations         │
│        + ## Further Reading with arXiv papers            │
└─────────────────────────────────────────────────────────┘
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
             Streamlit UI        CLI agent
              (app.py)     (mcp_client/agent.py)
```

---

## Prerequisites

- Python 3.10+
- [Pinecone](https://www.pinecone.io/) account (free serverless tier works)
- [Groq](https://console.groq.com/) API key (free tier)

---

## Installation

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd "RAG video project"

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy sentencizer (no model weights — rule-based only)
python -m spacy download en_core_web_sm
```

---

## Configuration

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=youtube-transcripts   # created automatically on first ingest
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3-32b                 # optional — this is the default
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # optional — this is the default
```

---

## Usage

### Ingest videos

```bash
# Single video
python ingest.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Multiple videos
python ingest.py "https://youtu.be/AAA" "https://youtu.be/BBB"

# From a file (one URL per line, # for comments)
python ingest.py --urls-file videos.txt
```

Re-ingesting the same video is safe — chunks are upserted by deterministic ID and overwrite the previous version.

### Streamlit UI

```bash
streamlit run app.py
```

Opens a browser with two tabs:
- **Ingest Videos** — paste URLs, click Ingest, see live per-video status
- **Chat** — ask questions, get streaming answers with citations and paper suggestions

### CLI agent

```bash
# Interactive mode
python mcp_client/agent.py

# Single question
python mcp_client/agent.py "What does the speaker say about transformers?"

# Show tool calls
python mcp_client/agent.py --show-tools "your question"
```

---

## Project structure

```
RAG video project/
├── app.py                      # Streamlit UI (ingest tab + chat tab)
├── ingest.py                   # CLI ingestion entrypoint
├── requirements.txt
│
├── ingestion/                  # Reusable ingestion library
│   ├── transcript.py           # Fetch transcript + metadata via yt-dlp
│   ├── chunker.py              # Timeframe-aware semantic chunking
│   ├── embedder.py             # BAAI/bge-small-en-v1.5 wrapper
│   └── store.py                # Pinecone upsert helpers
│
├── mcp_server/                 # MCP server (transcript search)
│   ├── server.py               # FastMCP server — exposes search_transcripts
│   └── retrieval.py            # Pinecone query + CrossEncoder reranking
│
└── mcp_client/                 # MCP client + agent
    └── agent.py                # LangGraph ReAct agent via MultiServerMCPClient
```

---

## How it works

### Timeframe-aware semantic chunking

Raw YouTube transcripts are a flat list of short segments (typically 2–5 seconds each), each with a `start` time and `duration`. The chunker groups these into semantically coherent chunks while preserving exact timestamps.

**Step 1 — Build the offset index**

All segment texts are concatenated into one string, and a sorted index of `(char_start, char_end, segment)` tuples is built alongside it. This lets any character position in the full text be mapped back to its source segment and timestamp.

**Step 2 — Sentence splitting**

spaCy's rule-based sentencizer splits the full text into sentences. For each sentence, its character span is binary-searched against the offset index to find all overlapping segments. The sentence's `start_time = min(seg.start)` and `end_time = max(seg.start + seg.duration)` across those segments.

> **Fallback for auto-generated transcripts:** Auto-generated captions have no punctuation, so the sentencizer treats the entire transcript as one sentence. In this case the chunker falls back to using the raw segments directly as atomic units — each segment already has its own timestamp.

**Step 3 — Semantic breakpoints**

Every sentence is embedded with `BAAI/bge-small-en-v1.5` (using the `document` prompt for asymmetric retrieval). Cosine similarity is computed between each consecutive pair of sentence embeddings (dot product, since embeddings are L2-normalised). Wherever the similarity falls below the 10th percentile, a topic-shift boundary is placed.

**Step 4 — Chunk assembly**

Sentences between breakpoints are joined into a single `Chunk`. Each chunk's `start_time` is taken from its first sentence and `end_time` from its last. A deep-link URL of the form `https://youtu.be/{video_id}?t={start_time}` is stored in metadata, pointing to the exact moment in the video.

### Metadata filtering

Every chunk stored in Pinecone carries the following metadata fields:

| Field | Type | Purpose |
|---|---|---|
| `text` | string | Raw chunk text for reranking and display |
| `video_id` | string | YouTube video ID |
| `video_title` | string | Human-readable title |
| `video_url` | string | Canonical watch URL |
| `timestamp_url` | string | Deep-link to exact moment (`?t=N`) |
| `start_time` | float | Seconds from video start |
| `end_time` | float | Seconds to chunk end |
| `chunk_index` | int | Position within this video |
| `total_chunks` | int | Total chunks for this video |

The `search_transcripts` tool accepts an optional `video_id` parameter. When provided, it is passed to Pinecone as a `$eq` metadata filter, restricting retrieval to a single video. The `start_time` and `end_time` float fields also support `$gte`/`$lte` range filters for time-bounded queries.

### CrossEncoder reranking

Pinecone's ANN search returns approximate nearest neighbours by embedding cosine similarity, which sometimes surfaces passages that share vocabulary with the query but aren't actually relevant. The retriever compensates with a two-stage approach:

1. **Oversample** — fetch `n_results × 4` candidates from Pinecone (capped at 100)
2. **Rerank** — score every `(query, chunk_text)` pair with `cross-encoder/ms-marco-MiniLM-L-6-v2`, a model that reads both the query and the passage together and produces a relevance score
3. **Trim** — sort by reranker score descending and return the top `n_results`

The CrossEncoder model is loaded once as a lazy singleton and reused across all requests.

### MCP architecture

The project uses the [Model Context Protocol](https://modelcontextprotocol.io/) to separate the retrieval infrastructure from the agent:

- **`mcp_server/server.py`** — a FastMCP stdio server that exposes `search_transcripts` as a tool. Runs as a subprocess managed by the client.
- **`mcp-simple-arxiv`** — an external MCP server that exposes arXiv paper search.
- **`langchain-mcp-adapters`** — `MultiServerMCPClient` connects to both servers, discovers their tools, and converts them into LangChain `StructuredTool` objects.
- **LangGraph `StateGraph`** — implements the ReAct agent loop: the LLM decides whether to call a tool or produce a final answer, and the loop continues until no more tool calls are made.

In the Streamlit app, the agent runs in a dedicated background thread with its own asyncio event loop (`AgentRunner`). Response chunks are passed to the main thread via a `queue.Queue` and streamed to the UI with `st.write_stream()`.

---

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `PINECONE_API_KEY` | ✅ | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | | `youtube-transcripts` | Pinecone index name |
| `PINECONE_CLOUD` | | `aws` | Cloud provider for index creation |
| `PINECONE_REGION` | | `us-east-1` | Region for index creation |
| `GROQ_API_KEY` | ✅ | — | Groq API key |
| `GROQ_MODEL` | | `qwen/qwen3-32b` | Groq model ID |
| `EMBEDDING_MODEL` | | `BAAI/bge-small-en-v1.5` | Sentence-transformers model |
| `RERANKER_MODEL` | | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker model |
