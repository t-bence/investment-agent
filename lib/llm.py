from os import getenv
from typing import Callable, Dict, List

from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from lib.prompts import ANALYST_PROMPT
from lib.state import State


def get_chatbot(tools: List[Tool]) -> Callable[[State], Dict[str, list]]:
    api_key = getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key not set!")

    # OpenRouter docs: https://openrouter.ai/docs/overview/models
    llm = ChatOpenAI(
        api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="anthropic/claude-3.5-haiku",
    )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State) -> Dict[str, list]:
        """Invoke the chatbot"""

        messages = [{"role": "system", "content": ANALYST_PROMPT}] + state["messages"]

        response = [llm_with_tools.invoke(messages)]
        return {"messages": response}

    return chatbot
