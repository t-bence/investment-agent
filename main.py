"""
This is the LangGraph tutorial code

spell-checker:ignore tavily
"""

import sqlite3
from typing import Annotated, TypedDict

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
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

connection = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(connection)
# memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

memory_config = {"configurable": {"thread_id": 1}}


def stream_graph_updates(user_input_: str):
    """Display the chatbot's response in real-time."""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input_}]},
        config={"thread_id": 1},
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


st.title("Investment Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage("Hello! I'm your investment assistant. How can I help you today?")
    ]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

# takes new input in chat box from user and invokes the graph
if prompt := st.chat_input():
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        response = graph.invoke(
            {"messages": [{"role": "user", "content": prompt}]}, config=memory_config
        )
        st.markdown(response["messages"][-1].content)
