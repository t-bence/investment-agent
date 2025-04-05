# import sqlite3

from langgraph.checkpoint.memory import InMemorySaver

# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

from lib.llm import CriticalModel, PositiveModel, SummaryModel
from lib.state import State


class InvestmentAgent:
    """An agent that can provide investment advice based on user input."""

    def __init__(self):

        graph_builder = StateGraph(State)

        graph_builder.add_node("supportive", PositiveModel())

        graph_builder.add_node("critic", CriticalModel())

        graph_builder.add_edge("supportive", "critic")

        graph_builder.add_node("summary", SummaryModel())
        graph_builder.add_edge("critic", "summary")
        graph_builder.set_entry_point("supportive")

        # connection = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        # memory = SqliteSaver(connection)
        memory = InMemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)

        self.memory_config = {"configurable": {"thread_id": 1}}

    def invoke(self, state: State) -> str:

        response = self.graph.invoke(
            state,
            config=self.memory_config,
        )
        return response["messages"][-1].content
