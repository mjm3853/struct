"""Agent state schema."""

from dataclasses import dataclass, field

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass
class State:
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
