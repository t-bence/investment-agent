"""
This is the LangGraph tutorial code

spell-checker:ignore tavily
"""

from os import getenv
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools import YahooFinanceNewsTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


class State(TypedDict):
    """State of the graph"""
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

yahoo_tool = YahooFinanceNewsTool()

tools = [yahoo_tool]

api_key = getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("API key not set!")

# OpenRouter docs: https://openrouter.ai/docs/overview/models
llm = ChatOpenAI(
  api_key=api_key,
  openai_api_base="https://openrouter.ai/api/v1",
  model_name="anthropic/claude-3.5-haiku"
)

llm_with_tools = llm.bind_tools(tools)

critic_llm = ChatOpenAI(
  api_key=api_key,
  openai_api_base="https://openrouter.ai/api/v1",
  model_name="anthropic/claude-3.5-haiku"
)

def chatbot(state: State):
    """Invoke the chatbot"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def critic(state: State):
    """Criticize the previous response and highlight potential drawbacks"""
    # Get the last message from the chatbot
    last_response = state["messages"][-1].content
    
    # Use a separate LLM for criticism
    criticism_prompt = f"""
    Critically analyze the following response:
    {last_response}
    
    Provide a detailed critique highlighting:
    1. Potential biases
    2. Incomplete or oversimplified information
    3. Possible limitations or drawbacks
    4. Suggest areas for improvement
    
    Your analysis should be constructive and objective.
    """
    
    criticism = critic_llm.invoke([
        {"role": "system", "content": "You are a critical analyst providing objective feedback."},
        {"role": "user", "content": criticism_prompt}
    ])
    
    return {"messages": [criticism]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("critic", critic)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Add edges to include the critic in the workflow
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", "critic")  # After chatbot response, add critic analysis
graph_builder.set_entry_point("chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input_: str):
    """Display the chatbot's response in real-time."""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input_}]},
        config={"thread_id": 1}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
