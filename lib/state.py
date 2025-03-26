from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    """State of the graph"""

    messages: Annotated[list, add_messages]
