"""
This is the LangGraph tutorial code

spell-checker:ignore tavily
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from lib.graph import InvestmentAgent

agent = InvestmentAgent()

st.title("Investment Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage("Hello! I'm your investment assistant. How can I help you today?")
    ]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

# takes new input in chat box from user and invokes the graph
if prompt := st.chat_input():
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        response = agent.invoke(prompt)
        for msg in response:
            st.markdown(msg)
            st.session_state.messages.append(AIMessage(content=msg))
