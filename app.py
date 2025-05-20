import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chatbot.graph_setup import build_graph

graph, config = build_graph()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("💬 RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "🧑 You" if isinstance(msg, HumanMessage) else "🤖 Bot"
    st.chat_message(role).write(msg.content)

if prompt := st.chat_input("Ask your question here..."):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    st.chat_message("🧑 You").write(prompt)

    response_container = st.chat_message("🤖 Bot")
    response_placeholder = response_container.empty()

    full_response = ""
    last_ai_content = None
    input_data = {"messages": st.session_state.messages}

    for step in graph.stream(input_data, stream_mode="values", config=config):
        ai_msgs = [m for m in step["messages"] if isinstance(m, AIMessage)]
        if ai_msgs:
            current_content = ai_msgs[-1].content
            if current_content != last_ai_content:
                full_response = current_content
                response_placeholder.markdown(full_response + "▌")
                last_ai_content = current_content

    ai_msg = AIMessage(content=full_response)
    st.session_state.messages.append(ai_msg)
    response_placeholder.markdown(full_response)
