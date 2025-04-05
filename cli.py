from lib.graph import InvestmentAgent

if __name__ == "__main__":
    agent = InvestmentAgent()

    initial_state = {
        "messages": [],
        "thesis": "I live in Hungary. I want to invest in all world ETFs to diversify my portfolio.",
        "positive": "",
        "negative": "",
    }

    message = agent.invoke(initial_state)
    print(message)
