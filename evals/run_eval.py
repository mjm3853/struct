"""Run evals against the struct agent via LangSmith.

Uploads results to LangSmith as an experiment, where you can compare runs
across model changes, prompt edits, or tool modifications.

Four evaluators:
  1. tool_selection — did the agent call the expected tools?
  2. argument_accuracy — did it pass the right ticker/params?
  3. response_terms — does the final answer mention expected terms?
  4. response_quality — LLM-as-judge (Haiku) scoring accuracy, completeness, clarity, data usage

Usage:
    uv run python evals/upload_dataset.py          # one-time: create dataset in LangSmith
    uv run python evals/run_eval.py                # run experiment (requires LangSmith)
    uv run python evals/run_eval.py --prefix v2    # tag experiment
    uv run python evals/run_eval.py --local        # run from local JSONL, no LangSmith needed
"""

import asyncio
import json
import warnings
from pathlib import Path

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402

from struct_agent.graph import graph  # noqa: E402

DATASET_FILE = Path(__file__).parent / "dataset.jsonl"


# ---------------------------------------------------------------------------
# Target: reshapes dataset inputs into the graph's expected state
# ---------------------------------------------------------------------------


async def target(inputs: dict) -> dict:
    result = await graph.ainvoke({"messages": [HumanMessage(content=inputs["question"])]})
    return result


# ---------------------------------------------------------------------------
# Evaluators — each receives outputs (graph state) and reference_outputs
# ---------------------------------------------------------------------------


def _extract_tool_calls(outputs: dict) -> list[dict]:
    """Pull tool calls from the message history in the graph output."""
    calls = []
    for msg in outputs.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append({"name": tc["name"], "args": tc["args"]})
    return calls


def tool_selection(outputs: dict, reference_outputs: dict) -> dict:
    """Check that the agent called at least the expected tools.

    Uses subset matching: every expected tool must appear in the called tools
    at least as many times as specified. Extra tool calls are allowed (the agent
    may reasonably gather additional context).
    """
    tool_calls = _extract_tool_calls(outputs)
    called = sorted(tc["name"] for tc in tool_calls)
    expected = sorted(reference_outputs.get("expected_tools", []))

    # Check each expected tool appears at least the required number of times
    from collections import Counter

    called_counts = Counter(called)
    expected_counts = Counter(expected)
    missing = []
    for tool_name, count in expected_counts.items():
        if called_counts.get(tool_name, 0) < count:
            missing.append(f"{tool_name} (need {count}, got {called_counts.get(tool_name, 0)})")

    if missing:
        return {
            "key": "tool_selection",
            "score": 0.0,
            "comment": f"missing: {missing}, called: {called}",
        }
    return {"key": "tool_selection", "score": 1.0}


def argument_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    """Check that tool arguments contain the expected key-value pairs."""
    expected_args = reference_outputs.get("expected_args", {})
    if not expected_args:
        return {"key": "argument_accuracy", "score": 1.0}

    tool_calls = _extract_tool_calls(outputs)
    for tool_name, expected_kv in expected_args.items():
        matching = [tc for tc in tool_calls if tc["name"] == tool_name]
        if not matching:
            return {
                "key": "argument_accuracy",
                "score": 0.0,
                "comment": f"{tool_name} not called",
            }
        found = any(
            all(
                str(tc["args"].get(k, "")).upper() == str(v).upper() for k, v in expected_kv.items()
            )
            for tc in matching
        )
        if not found:
            return {
                "key": "argument_accuracy",
                "score": 0.0,
                "comment": f"{tool_name} args mismatch: {matching}",
            }

    return {"key": "argument_accuracy", "score": 1.0}


def response_terms(outputs: dict, reference_outputs: dict) -> dict:
    """Check that the final response contains all expected terms."""
    expected_terms = reference_outputs.get("expected_in_response", [])
    if not expected_terms:
        return {"key": "response_terms", "score": 1.0}

    messages = outputs.get("messages", [])
    content = messages[-1].content.lower() if messages else ""
    missing = [t for t in expected_terms if t.lower() not in content]

    if missing:
        return {
            "key": "response_terms",
            "score": round(1 - len(missing) / len(expected_terms), 2),
            "comment": f"missing: {missing}",
        }
    return {"key": "response_terms", "score": 1.0}


def response_quality(inputs: dict, outputs: dict) -> dict:
    """LLM-as-judge evaluator for overall response quality.

    Scores the agent's response on a 0-1 scale across four criteria:
    accuracy, completeness, clarity, and appropriate use of data.
    """
    from langchain_anthropic import ChatAnthropic

    from struct_agent.settings import Settings

    messages = outputs.get("messages", [])
    if not messages:
        return {"key": "response_quality", "score": 0.0, "comment": "no response"}

    response_content = messages[-1].content
    question = inputs.get("question", "")

    judge_prompt = f"""\
You are evaluating a financial markets AI agent. The user asked a question and \
the agent responded using real market data tools (stock quotes, options chains, \
price history, institutional holders).

Rate the response on a scale of 0.0 to 1.0 based on these criteria:
- **Accuracy**: Does it correctly interpret and present the data it retrieved?
- **Completeness**: Does it address all parts of the user's question?
- **Clarity**: Is the response well-organized and easy to understand?
- **Data usage**: Does it cite specific numbers rather than speaking vaguely?

User question: {question}

Agent response:
{response_content}

Respond with ONLY a JSON object in this exact format, nothing else:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

    settings = Settings()
    judge = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=settings.anthropic_api_key)
    result = judge.invoke(judge_prompt)

    try:
        content = result.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)
        return {
            "key": "response_quality",
            "score": float(parsed["score"]),
            "comment": parsed.get("reason"),
        }
    except (json.JSONDecodeError, KeyError, IndexError):
        return {
            "key": "response_quality",
            "score": 0.5,
            "comment": f"judge parse error: {result.content[:100]}",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EVALUATORS = [tool_selection, argument_accuracy, response_terms, response_quality]


async def run_langsmith(prefix: str):
    """Run evals via LangSmith aevaluate — requires valid LANGSMITH_API_KEY."""
    from langsmith import aevaluate

    results = await aevaluate(
        target,
        data="struct-agent",
        evaluators=EVALUATORS,
        experiment_prefix=prefix,
        max_concurrency=2,
    )

    passed = 0
    total = 0
    async for result in results:
        total += 1
        scores = {r.key: r.score for r in result.get("evaluation_results", {}).get("results", [])}
        all_pass = all(s >= 0.7 for s in scores.values())
        if all_pass:
            passed += 1
        status = "\033[32mPASS\033[0m" if all_pass else "\033[31mFAIL\033[0m"
        q = result["run"].inputs.get("question", "?")[:60]
        print(f"  {status}  {scores}  {q}")

    print(f"\n{passed}/{total} passed")


async def run_local():
    """Run evals locally from dataset.jsonl — no LangSmith needed."""
    cases = [json.loads(line) for line in DATASET_FILE.read_text().strip().splitlines()]

    passed = 0
    total = len(cases)

    for i, case in enumerate(cases):
        print(f"\n[{i}] {case['input']}")
        outputs = await target({"question": case["input"]})
        ref = {
            "expected_tools": case["expected_tools"],
            "expected_args": case.get("expected_args", {}),
            "expected_in_response": case.get("expected_in_response", []),
        }

        scores = {}
        eval_inputs = {"question": case["input"]}
        for evaluator in EVALUATORS:
            import inspect

            params = inspect.signature(evaluator).parameters
            if "inputs" in params and "reference_outputs" not in params:
                result = evaluator(inputs=eval_inputs, outputs=outputs)
            else:
                result = evaluator(outputs=outputs, reference_outputs=ref)
            scores[result["key"]] = result["score"]
            if result.get("comment"):
                print(f"    {result['key']}: {result['comment']}")

        all_pass = all(s >= 0.7 for s in scores.values())
        if all_pass:
            passed += 1
        status = "\033[32mPASS\033[0m" if all_pass else "\033[31mFAIL\033[0m"
        tools = [tc["name"] for tc in _extract_tool_calls(outputs)]
        print(f"  {status}  {scores}  called={tools}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed ({passed / total * 100:.0f}%)")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run struct agent evals")
    parser.add_argument("--prefix", default="struct-eval", help="Experiment prefix in LangSmith")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run from local dataset.jsonl without LangSmith",
    )
    args = parser.parse_args()

    if args.local:
        await run_local()
    else:
        await run_langsmith(args.prefix)


if __name__ == "__main__":
    asyncio.run(main())
