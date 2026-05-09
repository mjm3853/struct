# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**struct** is a LangGraph ReAct agent that queries structured financial market data via **yfinance** (Yahoo Finance). Python 3.12, managed with **uv**. Observability and evals run through **LangSmith**.

## Commands

```bash
uv sync                                          # Install dependencies
uv run struct                                     # Interactive chat REPL
uv run langgraph dev                              # LangGraph Studio (visual debugger)
uv run ruff check src/ evals/                     # Lint
uv run ruff format src/ evals/                    # Format
uv run python evals/upload_dataset.py             # Upload eval dataset to LangSmith (one-time)
uv run python evals/push_prompts.py               # Push eval prompts to LangSmith
uv run python evals/run_eval.py                   # Run eval experiment
uv run python evals/run_eval.py --prefix v2       # Tag an experiment
uv run python evals/run_eval.py --local           # Run without LangSmith upload
```

## Architecture

Two-node **ReAct loop**: `call_model → [tool_calls?] → tools → call_model → ... → end`.

The package is `src/struct_agent/` (not `struct`, which shadows Python's stdlib module).

- **graph.py** — `StateGraph` with `agent` and `tools` nodes, conditional edge routing on `tool_calls`. Exports both `build_graph()` and a module-level `graph` (compiled) used by LangGraph Studio and evals.
- **tools.py** — Sync `@tool` functions formatting yfinance data for the LLM. Exports `ALL_TOOLS`. Currently: `get_stock_quote`, `get_option_chain`, `get_price_history`, `get_institutional_holders`. `get_option_chain` auto-selects the nearest expiration when none is provided.
- **client.py** — Data layer wrapping `yfinance`. Pure functions returning dataclasses (`get_quote`, `get_option_chain`, `get_option_expirations`, `get_history`, `get_institutional_holders`). No API key required. Uses `_safe_int`/`_safe_float` helpers to handle NaN values in yfinance data.
- **settings.py** — `pydantic-settings` loading from `.env` with `extra="ignore"` so LangSmith env vars don't cause validation errors.
- **prompts.py** — System prompt with `{system_time}` placeholder for temporal grounding.
- **state.py** — `State` dataclass with `messages` field using LangGraph's `add_messages` reducer.
- **__main__.py** — Typer CLI, async REPL with rich markdown rendering.

## Adding a New Tool

1. Add dataclass(es) and a function in `client.py`.
2. Add a `@tool` function in `tools.py` and append to `ALL_TOOLS`.
3. Add eval cases to `evals/dataset.jsonl` and re-run `upload_dataset.py`.

No changes to the graph required.

## Evals

The `evals/` directory contains a LangSmith-integrated eval harness. See `evals/README.md` for the full workflow.

Four scoring dimensions: **tool_selection** (right tools called, subset matching), **argument_accuracy** (right ticker/params passed), **response_terms** (expected keywords in final answer), and **response_quality** (LLM-as-judge via Haiku scoring accuracy, completeness, clarity, data usage). A case passes when all scores are ≥ 0.7.

The **response_quality** judge prompt is managed in LangSmith as **`struct-judge-response-quality`**. Edit it in the LangSmith UI to iterate without code changes. Push local edits with `push_prompts.py`. The evaluator falls back to a local prompt if LangSmith is unreachable.

The dataset lives in `evals/dataset.jsonl` (JSONL, one case per line). The `upload_dataset.py` script pushes it to LangSmith as the **"struct-agent"** dataset.

## Environment

Requires `.env` with `ANTHROPIC_API_KEY`. For LangSmith tracing and evals, also set `LANGSMITH_API_KEY`, `LANGCHAIN_TRACING_V2=true`, and `LANGCHAIN_PROJECT`. See `.env.example`.
