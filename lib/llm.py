from os import getenv
from typing import Any, Dict, List

from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from lib.state import State


class BaseModel:
    """A class that represents a language model."""

    def __init__(self, tools: List[Tool] = None):
        self.tools = tools
        api_key = getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("API key not set!")

        # OpenRouter docs: https://openrouter.ai/docs/overview/models
        self.llm = ChatOpenAI(
            api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="anthropic/claude-3.5-haiku",
        )

        if tools:
            self.llm = self.llm.bind_tools(tools)

    def __call__(self, state: State) -> Dict[str, Any]:
        raise NotImplementedError


class PositiveModel(BaseModel):
    """A class that represents a positive language model."""

    def __init__(self):
        super().__init__()

    def __call__(self, state: State) -> Dict[str, Any]:
        prompt = f"""You are a financial analyst.
        Your task is to provide positive feedback on the given investment opportunity.
        The feedback should be concise and highlight the strengths of the investment.
        The investment thesis is the following:
        
        {state['thesis']}
        """

        return {"positive": self.llm.invoke(prompt).content}


class CriticalModel(BaseModel):
    """A class that represents a critical language model."""

    def __init__(self):
        super().__init__()

    def __call__(self, state: State) -> Dict[str, Any]:
        prompt = f"""You are a financial analyst.
        You are tasked to provide critical feedback on the given investment opportunity.
        The feedback should be concise and highlight the weaknesses of the investment.
        The investment thesis is the following:
        
        {state['thesis']}
        
        The supporting evidence for this investment is as follows:
        {state['positive']}
        
        Be critical but fair.
        """

        return {"negative": self.llm.invoke(prompt).content}


class SummaryModel(BaseModel):
    """A class that represents a summarizing language model."""

    def __init__(self):
        super().__init__()

    def __call__(self, state: State) -> Dict[str, Any]:
        prompt = f"""
        Read this investment thesis, the positives and the negatives,
        and summarize them in a concise manner. Use markdown formatting!
        
        Thesis:
        {state['thesis']}
                
        Positives:
        {state['positive']}
        
        Negatives:
        {state['negative']}
        """

        response = self.llm.invoke(prompt).content

        return {"messages": [{"role": "assistant", "content": response}]}
