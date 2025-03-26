"""
This is the LangGraph tutorial code

spell-checker:ignore tavily
"""

# import sqlite3

from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver

# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lib.llm import get_chatbot
from lib.state import State
from lib.tools import get_tools

tools = get_tools()

chatbot = get_chatbot(tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# connection = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
# memory = SqliteSaver(connection)
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input_: str):
    """Display the chatbot's response in real-time."""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input_}]},
        config={"thread_id": 1},
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            # connection.close()
            break

        stream_graph_updates(user_input)
