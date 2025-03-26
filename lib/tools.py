import yfinance as yf
from langchain.tools import Tool
from langchain_community.tools import YahooFinanceNewsTool

# tavily_tool = TavilySearchResults(max_results=2) # needs API key

# yahoo_tool = YahooFinanceNewsTool() # does not work


# Define a function to retrieve stock price
def get_stock_price(symbol: str) -> str:
    """Retrieve the current stock price for a given stock symbol."""
    try:
        ticker = yf.Ticker(symbol)
        # print(ticker.fast_info.last_price)
        current_price = ticker.info["currentPrice"]
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
