"""Push eval prompts to LangSmith for versioned management.

Run this whenever you update a prompt locally and want to sync it to LangSmith.
Edit prompts in the LangSmith UI, then pull them at eval time — no code change needed.

Usage:
    uv run python evals/push_prompts.py
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

load_dotenv()

PROMPTS = {
    "struct-judge-response-quality": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are evaluating a financial markets AI agent. The user asked a "
                "question and the agent responded using real market data tools (stock "
                "quotes, options chains, price history, institutional holders).\n\n"
                "Rate the response on a scale of 0.0 to 1.0 based on these criteria:\n"
                "- **Accuracy**: Does it correctly interpret and present the data it retrieved?\n"
                "- **Completeness**: Does it address all parts of the user's question?\n"
                "- **Clarity**: Is the response well-organized and easy to understand?\n"
                "- **Data usage**: Does it cite specific numbers rather than speaking vaguely?\n\n"
                "Respond with ONLY a JSON object in this exact format, nothing else:\n"
                '{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}',
            ),
            (
                "human",
                "User question: {question}\n\nAgent response:\n{response}",
            ),
        ]
    ),
}


def main():
    client = Client()
    for name, prompt in PROMPTS.items():
        url = client.push_prompt(name, object=prompt)
        print(f"Pushed '{name}' -> {url}")


if __name__ == "__main__":
    main()
