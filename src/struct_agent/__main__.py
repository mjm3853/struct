"""CLI entry point for the struct agent."""

import asyncio

import typer
from rich.console import Console
from rich.markdown import Markdown

from struct_agent.graph import graph as agent_graph

app = typer.Typer(help="Chat with a financial markets agent powered by yfinance data.")
console = Console()


async def _chat_loop():
    messages = []

    console.print("[bold]struct[/bold] — financial markets agent")
    console.print("Type your question, or 'quit' to exit.\n")

    while True:
        try:
            user_input = console.input("[bold green]you:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})
        result = await agent_graph.ainvoke({"messages": messages})
        messages = result["messages"]

        last = messages[-1]
        console.print()
        console.print("[bold blue]agent:[/bold blue]")
        console.print(Markdown(last.content))
        console.print()


@app.command()
def chat():
    """Start an interactive chat session with the agent."""
    asyncio.run(_chat_loop())


if __name__ == "__main__":
    app()
