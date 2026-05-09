# struct

A LangGraph agent for querying structured financial market data. Built on [yfinance](https://github.com/ranaroussi/yfinance) for free, real-time stock and options data — no paid API keys required for market data.

## Quickstart

```bash
uv sync                          # install dependencies
cp .env.example .env             # add your ANTHROPIC_API_KEY
uv run struct                    # start the chat REPL
```

Or run in LangGraph Studio for a visual debugger:

```bash
uv run langgraph dev
```

## What It Can Do

Ask natural language questions about financial markets. The agent picks the right tools automatically:

- **Stock quotes** — price, volume, market cap, P/E, 52-week range
- **Options chains** — strikes, bid/ask, volume, open interest, implied volatility
- **Price history** — OHLCV bars at any interval (1m to 3mo) over any period (1d to max)
- **Institutional holders** — top shareholders, share counts, % of float

Example queries:
- "What's AAPL trading at?"
- "Show me the most active call options on TSLA"
- "How has NVDA performed over the last 6 months?"
- "Compare P/E ratios of AAPL and GOOGL"
- "Who are the biggest institutional holders of MSFT?"

## Architecture

The agent is a two-node **ReAct loop** built with LangGraph:

```
user message → call_model → [tool_calls?] → tools → call_model → ... → response
```

The LLM decides which tools to call and synthesizes the results into a natural language answer. All market data comes from Yahoo Finance via yfinance (15-20 min delay during market hours).

## Evals

The project includes a LangSmith-integrated eval harness with four scoring dimensions: tool selection, argument accuracy, response terms, and an LLM-as-judge response quality score. The judge prompt is managed in LangSmith for no-code iteration. See [evals/README.md](evals/README.md) for the full workflow.

## Environment

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | Powers the LLM (Claude) |
| `LANGSMITH_API_KEY` | For evals | LangSmith experiment tracking + prompt management |
| `LANGCHAIN_TRACING_V2` | For tracing | Set to `true` to enable |
| `LANGCHAIN_PROJECT` | For tracing | LangSmith project name |
