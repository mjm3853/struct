# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**struct** is an experimental LangGraph agent that interacts with structured financial market data via **yfinance** (Yahoo Finance). Python 3.12, managed with **uv**.

## Commands

```bash
uv sync                      # Install dependencies
uv run struct                # Launch interactive chat REPL
uv run langgraph dev         # Run in LangGraph Studio (uses langgraph.json)
uv run ruff check src/       # Lint
uv run ruff format src/      # Format
```

## Architecture

The agent follows a **ReAct loop**: `call_model → [tool_calls?] → tools → call_model → ... → end`.

The package lives in `src/struct_agent/` (not `struct`, to avoid shadowing Python's stdlib `struct` module).

- **graph.py** — LangGraph `StateGraph` with two nodes (`agent`, `tools`) and a conditional edge. Compiles to a `CompiledStateGraph`.
- **tools.py** — `@tool`-decorated sync functions that format yfinance data for the LLM. Exports `ALL_TOOLS` list.
- **client.py** — Data layer wrapping `yfinance`. Pure functions returning dataclasses: `get_quote`, `get_option_chain`, `get_history`, `get_institutional_holders`. No API key required.
- **settings.py** — `pydantic-settings` config loading `ANTHROPIC_API_KEY` from `.env`.
- **prompts.py** — System prompt with `{system_time}` placeholder.
- **__main__.py** — Typer CLI with a `chat` command running an async REPL.

## Adding a New Tool

1. Add dataclass and function in `client.py`.
2. Add `@tool` function in `tools.py` and append to `ALL_TOOLS`.

No changes to the graph needed.

## Environment

Requires a `.env` file with `ANTHROPIC_API_KEY`. See `.env.example`. No API key needed for market data (yfinance uses public Yahoo Finance endpoints).
