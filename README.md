# Investment agent

This is an experiment with LangGraph to create an agent that can answer questions about investments.

To run the agent with UI, use the following command:

```[bash]
streamlit run investment-agent/gui.py
```

## Further ideas

Create separate LLMs. There should be one positive and negative LLM. The positive LLM will answer questions about investments, while the negative LLM will answer questions about risks. Then there should be one that summarizes the concerns and gives a balanced response.

## Current issues

- There is no streaming.
- The CLI works by displaying the last message, but I'm not sure it would work with interactive chat. 
- The UI is not tested.
