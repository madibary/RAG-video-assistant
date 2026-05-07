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
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import MessagesState

from mcp_client.core import MODEL_NAME, MCP_SERVERS, build_agent

load_dotenv(_PROJECT_ROOT / ".env")


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

    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    agent = build_agent(llm, tools)

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
