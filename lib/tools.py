import os

import yfinance as yf
from langchain.tools import Tool
from langchain_community.tools import YahooFinanceNewsTool

# tavily_tool = TavilySearchResults(max_results=2) # needs API key

# yahoo_tool = YahooFinanceNewsTool() # does not work

os.environ["USER_AGENT"] = "AI Investment Agent"


# Define a function to retrieve stock price
def get_stock_price(symbol: str) -> str:
    """Retrieve the current stock price for a given stock symbol."""
    try:
        ticker = yf.Ticker(symbol)
        current_price = (
            ticker.info["currentPrice"]
            if "currentPrice" in ticker.info.keys()
            else ticker.info["regularMarketPrice"]  # needed for ETFs
        )
        return f"The current price of {symbol} is ${current_price:.2f}"
    except Exception as e:
        return f"Error retrieving stock price for {symbol}: {str(e)}"


# Create a Tool for stock price retrieval
stock_price_tool = Tool(
    name="get_stock_price",
    func=get_stock_price,
    description="Retrieves the current stock price for a given stock symbol",
)


def get_tools() -> list[Tool]:
    return [stock_price_tool]


if __name__ == "__main__":
    print(get_stock_price("CSPX.L"))
