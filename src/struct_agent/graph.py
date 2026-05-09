"""LangGraph ReAct agent graph."""

from datetime import datetime, timezone

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from struct_agent.prompts import SYSTEM_PROMPT
from struct_agent.settings import Settings
from struct_agent.state import State
from struct_agent.tools import ALL_TOOLS


async def call_model(state: State) -> dict:
    from langchain_anthropic import ChatAnthropic

    settings = Settings()
    model = ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,
    ).bind_tools(ALL_TOOLS)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    system = SYSTEM_PROMPT.format(system_time=now)
    messages = [SystemMessage(content=system)] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: State) -> str:
    last = state.messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": "__end__"})
    graph.add_edge("tools", "agent")
    return graph.compile()


# Module-level compiled graph for langgraph dev / Studio
graph = build_graph()
