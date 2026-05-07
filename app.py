#!/usr/bin/env python3
"""
app.py — Streamlit UI for the YouTube RAG pipeline.

Two tabs:
  Ingest  — paste YouTube URLs, see live status per video
  Chat    — ask questions, get streaming answers with video citations + papers

Run:
  streamlit run app.py
"""

import asyncio
import os
import queue
import sys
import threading
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="YouTube RAG", page_icon="🎬", layout="wide")

st.markdown("""
<style>
/* Hide Streamlit chrome to reclaim vertical space */
[data-testid="stHeader"]  { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
footer                    { display: none !important; }
#MainMenu                 { display: none !important; }

/* Remove default top/bottom padding from the main content block */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 0   !important;
}

/*
 * Streamlit sets the height inline on the first child div inside
 * stVerticalBlockBorderWrapper, not on the wrapper itself.
 * Target both the wrapper and its first child to override reliably.
 * The offset accounts for: title + tab bar + chat input + padding ≈ 160px.
 */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
    height: calc(100vh - 160px) !important;
    max-height: calc(100vh - 160px) !important;
    overflow-y: auto !important;
}
</style>
""", unsafe_allow_html=True)

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import Connection, MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ingest import process_video
from ingestion.embedder import get_dimension
from ingestion.store import get_client as get_pinecone_client
from ingestion.store import get_or_create_index

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("GROQ_MODEL", "qwen/qwen3-32b")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "youtube-transcripts")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about YouTube video content \
that has been indexed into a knowledge base.

## Step 1 — Answer from video transcripts
Call search_transcripts one or more times to find relevant passages.
Try different phrasings if the first search does not return useful results.
Base your answer only on what is in the retrieved passages.
Each result contains a "citation" field — include it verbatim after every claim:
    <claim text> [Video Title – H:MM:SS](timestamp_url)
If you cannot find the answer, say so clearly.

## Step 2 — Further Reading (academic papers only)
After writing your answer, call search_papers ONCE with the core topic.
This tool returns academic papers from Semantic Scholar — not videos.
Present the results under a "## Further Reading" heading:
    - [Paper Title](url) — Author A, Author B (Year)
      > One-sentence summary of what the paper is about.
Include 2–3 papers. If search_papers returns no results, omit this section.
IMPORTANT: Never put video citations or timestamps in the Further Reading section.\
"""

_MCP_SERVERS: dict[str, Connection] = {
    "youtube-rag": StdioConnection(
        transport="stdio",
        command=sys.executable,
        args=[str(_PROJECT_ROOT / "mcp_server" / "server.py")],
        env=dict(os.environ),
    ),
    "arxiv": StdioConnection(
        transport="stdio",
        command=sys.executable,
        args=["-m", "mcp_simple_arxiv"],
        env=dict(os.environ),
    ),
}

# ---------------------------------------------------------------------------
# Agent runner — lives in a background thread with its own event loop
# ---------------------------------------------------------------------------


def _build_agent(llm, tools):
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def _call_model(state: MessagesState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}

    def _should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", _call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


class AgentRunner:
    """Wraps the LangGraph agent + MCP sessions in a dedicated background thread.

    Streamlit reruns the script on every interaction, so the runner is cached
    with @st.cache_resource and shared across all sessions for the lifetime of
    the server process.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._agent = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error:
            raise self._error

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._setup())
            self._loop.run_forever()
        except Exception as exc:
            self._error = exc
            self._ready.set()

    async def _setup(self) -> None:
        llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            model=MODEL_NAME,
            temperature=0,
        )
        client = MultiServerMCPClient(_MCP_SERVERS)
        tools = await client.get_tools()
        self._agent = _build_agent(llm, tools)
        self._ready.set()

    def ask(self, question: str):
        """Sync generator that yields text chunks — compatible with st.write_stream()."""
        q: queue.Queue[str | None] = queue.Queue()

        async def _collect() -> None:
            inputs = MessagesState(messages=[HumanMessage(content=question)])
            try:
                async for chunk, _ in self._agent.astream(inputs, stream_mode="messages"):
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        q.put(chunk.content)
            finally:
                q.put(None)

        asyncio.run_coroutine_threadsafe(_collect(), self._loop)

        while True:
            item = q.get()
            if item is None:
                break
            yield item


@st.cache_resource(show_spinner="Starting agent…")
def get_runner() -> AgentRunner:
    return AgentRunner()


@st.cache_resource(show_spinner="Connecting to Pinecone…")
def get_index():
    pc = get_pinecone_client()
    return get_or_create_index(
        pc,
        index_name=INDEX_NAME,
        dimension=get_dimension(EMBEDDING_MODEL),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎬 Video Research Assistant")

ingest_tab, chat_tab = st.tabs(["Ingest Videos", "Chat"])

# ── Ingest ────────────────────────────────────────────────────────────────────
with ingest_tab:
    st.markdown("Paste one YouTube URL per line, then click **Ingest**.")

    raw = st.text_area(
        "YouTube URLs",
        height=180,
        placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
    )

    if st.button("Ingest", type="primary"):
        urls = [
            u.strip()
            for u in raw.splitlines()
            if u.strip() and not u.strip().startswith("#")
        ]

        if not urls:
            st.warning("No URLs entered.")
        else:
            try:
                index = get_index()
            except Exception as exc:
                st.error(f"Could not connect to Pinecone: {exc}")
                st.stop()

            for url in urls:
                with st.status(f"`{url}`", expanded=True) as status:
                    result = process_video(
                        url=url,
                        index=index,
                        model_name=EMBEDDING_MODEL,
                        breakpoint_percentile=10,
                        verbose=False,
                    )
                    if result["status"] == "ok":
                        status.update(
                            label=f"✅ **{result['title']}** — {result['chunks']} chunks stored",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status.update(
                            label=f"❌ {result['error']}",
                            state="error",
                        )

# ── Chat ──────────────────────────────────────────────────────────────────────
with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Fixed-height scrollable container keeps the input anchored below it
    messages_container = st.container(height=300, border=False)
    with messages_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about your videos…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with messages_container:
            with st.chat_message("user"):
                st.markdown(question)

        try:
            runner = get_runner()
        except Exception as exc:
            st.error(f"Agent failed to start: {exc}")
            st.stop()

        with messages_container:
            with st.chat_message("assistant"):
                response = st.write_stream(runner.ask(question))

        st.session_state.messages.append({"role": "assistant", "content": response})
