#!/usr/bin/env python3
"""
mcp_client/agent.py — Agentic RAG Q&A via two MCP servers.

Uses MultiServerMCPClient from langchain-mcp-adapters to connect to both
servers and get LangChain tools directly — no manual MCP→LangChain conversion
needed.

Servers:
  mcp_server/server.py    — YouTube transcript search + academic paper search
  papers_server/server.py — Semantic Scholar paper search

Usage:
  python mcp_client/agent.py "What does the speaker say about neural networks?"
  python mcp_client/agent.py          # interactive mode
  python mcp_client/agent.py --show-tools "..."
  python -m mcp_client

Requires:
  GROQ_API_KEY     in .env or environment
  PINECONE_API_KEY in .env or environment
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient, Connection
from langchain_mcp_adapters.sessions import StdioConnection
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("GROQ_MODEL", "qwen/qwen3-32b")

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
# System prompt
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# LangGraph agent factory
# ---------------------------------------------------------------------------


def _build_agent(llm: ChatGroq, tools: list):
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


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


async def run_session(
    questions: list[str] | None = None,
    show_tool_calls: bool = False,
) -> None:
    if not os.environ.get("GROQ_API_KEY"):
        print(
            "Error: GROQ_API_KEY is not set.\n"
            "Get a free key at https://console.groq.com/keys and add it to .env",
            file=sys.stderr,
        )
        sys.exit(1)

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY", ""),
        model=MODEL_NAME,
        temperature=0,
    )

    client = MultiServerMCPClient(_MCP_SERVERS)
    tools = await client.get_tools()
    agent = _build_agent(llm, tools)

    if questions:
        for question in questions:
            await _ask_once(agent, question, show_tool_calls)
    else:
        print(f"YouTube RAG Agent (MCP)  [{MODEL_NAME}]")
        print("Type 'quit' to exit  |  prefix with --show-tools to see searches\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            verbose = show_tool_calls or question.startswith("--show-tools ")
            question = question.removeprefix("--show-tools ").strip()

            print("\nAgent: ", end="", flush=True)
            await _ask_once(agent, question, verbose)
            print()


async def _ask_once(agent, question: str, show_tool_calls: bool) -> None:
    inputs = MessagesState(messages=[HumanMessage(content=question)])

    async for chunk, _ in agent.astream(inputs, stream_mode="messages"):
        if show_tool_calls and isinstance(chunk, ToolMessage):
            try:
                results = json.loads(chunk.content)
                print(
                    f"\n[searched: {len(results)} result(s) for tool call "
                    f"{chunk.tool_call_id[:8]}…]",
                    flush=True,
                )
            except (json.JSONDecodeError, TypeError):
                pass

        if isinstance(chunk, AIMessageChunk) and chunk.content:
            print(chunk.content, end="", flush=True)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    show_tools = "--show-tools" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        asyncio.run(run_session(questions=[" ".join(args)], show_tool_calls=show_tools))
    else:
        asyncio.run(run_session(show_tool_calls=show_tools))


if __name__ == "__main__":
    main()
