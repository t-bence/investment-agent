from lib.graph import InvestmentAgent

if __name__ == "__main__":
    agent = InvestmentAgent()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        messages = agent.invoke(user_input)
        for message in messages:
            print(f"Assistant: {message}")
