# Evals

Evaluation harness for the struct agent, built on [LangSmith's `aevaluate()`](https://docs.langchain.com/langsmith/evaluate-graph). Each eval run becomes a **LangSmith experiment** that you can compare across model swaps, prompt changes, or tool modifications.

## Setup

Make sure your `.env` contains valid keys:

```
ANTHROPIC_API_KEY=...
LANGSMITH_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=yfinance
```

Upload the dataset to LangSmith (one-time):

```bash
uv run python evals/upload_dataset.py
```

This creates a dataset called **"struct-agent"** in your LangSmith workspace.

## Running Experiments

```bash
uv run python evals/run_eval.py                        # default experiment
uv run python evals/run_eval.py --prefix sonnet-v2      # tag the experiment
uv run python evals/run_eval.py --local                  # run without uploading to LangSmith
```

Each run invokes the compiled graph against every example in the dataset, applies the evaluators, and uploads the results as a named experiment. Open LangSmith to compare experiments side-by-side.

## Evaluators

Four dimensions are scored per example, each returning a 0.0–1.0 score:

**tool_selection** — Did the agent call at least the expected tools? Uses subset matching — every expected tool must appear at least the required number of times, but extra calls are allowed (the agent may reasonably gather additional context).

**argument_accuracy** — Did the tool calls contain the right key-value pairs? Only checks fields explicitly specified in `expected_args`; extra arguments the agent passes are ignored. Case-insensitive matching on values.

**response_terms** — Does the agent's final text response contain all expected terms? Partial credit is given proportional to how many terms matched. For example, if 2 of 3 expected terms appear, the score is 0.67.

**response_quality** — LLM-as-judge evaluator using Claude Haiku. Scores the agent's response on accuracy (correct interpretation of data), completeness (addresses all parts of the question), clarity (well-organized and readable), and data usage (cites specific numbers rather than speaking vaguely). This evaluator does not use reference outputs — it judges the response against the original question directly.

## Dataset Format

The dataset lives in `dataset.jsonl`. Each line is a JSON object:

```json
{
  "input": "What's Apple's current stock price?",
  "expected_tools": ["get_stock_quote"],
  "expected_args": {"get_stock_quote": {"ticker": "AAPL"}},
  "expected_in_response": ["AAPL", "Price"]
}
```

- **input** — The user question sent to the agent.
- **expected_tools** — Tool names the agent should call (duplicates allowed for multi-call scenarios).
- **expected_args** — Subset of arguments to verify per tool. Empty `{}` means "don't check args."
- **expected_in_response** — Terms the final response must contain (case-insensitive).

To add new cases, append a line to `dataset.jsonl` and re-run `upload_dataset.py`.

## Typical Workflow

1. Edit the prompt, swap the model, or add a tool.
2. Run `uv run python evals/run_eval.py --prefix my-change`.
3. Open LangSmith, compare the new experiment against the baseline.
4. Drill into failures to see the full trace (tool calls, arguments, latencies).
