import sqlite3

from langgraph.checkpoint.memory import InMemorySaver

# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lib.llm import get_chatbot
from lib.state import State
from lib.tools import get_tools


class InvestmentAgent:
    """An agent that can provide investment advice based on user input."""

    def __init__(self):
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
        # memory = InMemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)

        self.memory_config = {"configurable": {"thread_id": 1}}

    def invoke(self, prompt: str) -> list[str]:
        # Implement the logic to handle the prompt and return a response
        response = self.graph.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=self.memory_config,
        )
        return [msg.content for msg in response["messages"]]
