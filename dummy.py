from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
import os

# Define our state
class AgentState(TypedDict):
    ticker: str
    thesis: str
    messages: List[Dict[str, Any]]
    market_data: Dict[str, Any]
    critique: str

# Define nodes for our graph
def retrieve_market_data(state: AgentState) -> AgentState:
    """Node that retrieves financial data and news in one step"""
    ticker = state["ticker"]
    
    # In a real implementation, this would call financial APIs
    # For example, using yfinance, alpha_vantage, or a news API
    
    # Simple mock implementation
    market_data = {
        "financial": {
            "current_price": 150.25,
            "pe_ratio": 22.4,
            "market_cap": "1.2T",
        },
        "news": [
            {"title": f"Analysts discuss {ticker}'s market position", "date": "2025-03-15"},
            {"title": f"Industry trends affecting {ticker}", "date": "2025-03-10"},
        ]
    }
    
    return {"market_data": market_data}

def generate_critique(state: AgentState) -> AgentState:
    """Node that generates investment critique based on retrieved data"""
    from langchain_openai import ChatOpenAI
    
    # In a real implementation, use your preferred LLM
    llm = ChatOpenAI(temperature=0.7)
    
    prompt = f"""
    Based on the following investment thesis and market data, provide a thoughtful critique:
    
    TICKER: {state["ticker"]}
    
    INVESTOR'S THESIS: {state["thesis"]}
    
    MARKET DATA:
    - Current Price: {state["market_data"]["financial"]["current_price"]}
    - P/E Ratio: {state["market_data"]["financial"]["pe_ratio"]}
    - Market Cap: {state["market_data"]["financial"]["market_cap"]}
    
    RECENT NEWS:
    {state["market_data"]["news"][0]["title"]} ({state["market_data"]["news"][0]["date"]})
    {state["market_data"]["news"][1]["title"]} ({state["market_data"]["news"][1]["date"]})
    
    Provide a balanced critique that covers:
    1. Potential flaws in the investment reasoning
    2. Key risks not mentioned in the thesis
    3. Alternative interpretations of the available data
    4. Suggestions for improving the investment analysis
    
    Be direct and specific in your critique.
    """
    
    critique = llm.invoke(prompt).content
    
    return {"critique": critique}

def format_response(state: AgentState) -> AgentState:
    """Node that formats the final response"""
    response = f"""
    # Investment Critique: {state["ticker"]}
    
    I've analyzed your investment thesis for {state["ticker"]} and found some important points to consider:
    
    {state["critique"]}
    
    This critique is based on current market data and news as of today. Remember that markets change quickly, and you should verify all information before making investment decisions.
    """
    
    return {"messages": state["messages"] + [{"role": "assistant", "content": response}]}

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve_market_data", retrieve_market_data)
workflow.add_node("generate_critique", generate_critique)
workflow.add_node("format_response", format_response)

# Define edges
workflow.add_edge("retrieve_market_data", "generate_critique")
workflow.add_edge("generate_critique", "format_response")
workflow.add_edge("format_response", END)

# Set the entry point
workflow.set_entry_point("retrieve_market_data")

# Compile the graph
app = workflow.compile()

# Example usage
def get_investment_critique(ticker: str, thesis: str):
    """Run the investment critique agent with LangGraph"""
    # Initialize the state
    initial_state = {
        "ticker": ticker,
        "thesis": thesis,
        "messages": [],
        "market_data": {},
        "critique": ""
    }
    
    # Execute the graph
    result = app.invoke(initial_state)
    
    # Return the final message
    return result["messages"][-1]["content"]

# Example of how to use it
if __name__ == "__main__":
    ticker = "AAPL"
    thesis = """
    I believe Apple is a good investment because they have strong brand loyalty,
    consistent revenue from services, and they might release an AI product soon
    that will boost their stock price.
    """
    
    critique = get_investment_critique(ticker, thesis)
    print(critique)
