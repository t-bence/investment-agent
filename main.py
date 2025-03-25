"""
This is the LangGraph tutorial code

spell-checker:ignore tavily
"""

from os import getenv
from typing import Annotated, TypedDict
import yfinance as yf
from langchain.tools import Tool

from langchain_openai import ChatOpenAI
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import YahooFinanceNewsTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State of the graph"""
    messages: Annotated[list, add_messages]

# tavily_tool = TavilySearchResults(max_results=2)

yahoo_tool = YahooFinanceNewsTool()


# Define a function to retrieve stock price
def get_stock_price(symbol: str) -> str:
    """Retrieve the current stock price for a given stock symbol."""
    try:
        ticker = yf.Ticker(symbol)
        current_price = ticker.info['currentPrice']
        return f"The current price of {symbol} is ${current_price:.2f}"
    except Exception as e:
        return f"Error retrieving stock price for {symbol}: {str(e)}"

# Create a Tool for stock price retrieval
stock_price_tool = Tool(
    name="get_stock_price",
    func=get_stock_price,
    description="Retrieves the current stock price for a given stock symbol"
)

# Update the tools list to include the new stock price tool
tools = [stock_price_tool]

api_key = getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("API key not set!")

# OpenRouter docs: https://openrouter.ai/docs/overview/models
llm = ChatOpenAI(
  api_key=api_key,
  openai_api_base="https://openrouter.ai/api/v1",
  model_name="anthropic/claude-3.5-haiku" #"mistralai/mistral-small-3.1-24b-instruct:free"
)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    """Invoke the chatbot"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

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
