import streamlit as st

import appl
from appl import AIMessage, Conversation, UserMessage, gen, ppl

appl.init(settings={"tracing": {"enabled": False}})
# TODO: fix the incompatibility between streamlit and tracing, temporarily disable tracing

with st.sidebar:
    model_name = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "claude-35-sonnet"],
        index=1,
    )
    api_key = st.text_input(
        "Your API Key for the selected model", key="api_key", type="password"
    )
    if api_key == "":
        api_key = None
    "[View the source code](https://github.com/appl-team/appl/blob/main/examples/usage/streamlit_app.py)"
    "Built with [APPL](https://appl-team.github.io/appl/) and [Streamlit](https://streamlit.io/)"


@ppl
def chat(conversation: Conversation):
    conversation
    return gen(model_name, api_key=api_key, stream=True)


st.title("🤖 ChatBot")
# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = Conversation()

history = st.session_state.history
# Display chat messages from history on app rerun
for message in history:
    with st.chat_message(message.role.type):
        st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    history.append(UserMessage(prompt))

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    response = chat(history)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response.text_stream)
    history.append(AIMessage(response))
