import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage


def initialize_session():
    """ Initialize session state to store chat history"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """ Display previous messages from session history """
    for msg in st.session_state.messages:
        role = "You" if isinstance(msg, HumanMessage) else "Bot"
        st.chat_message(role).write(msg.content)

def handle_user_input():
    """ Capture and store user input """
    if prompt := st.chat_input("Ask your question"):
        user_msg = HumanMessage(content=prompt)
        st.session_state.messages.append(user_msg)
        st.chat_message("You").write(prompt)
        return user_msg
    return None

def stream_response(graph, config):
    """ Stream AI response and update UI """
    response_container = st.chat_message("Bot")
    response_placeholder = response_container.empty()

    full_response = ""
    last_ai_content = None
    st.session_state.messages.append(AIMessage(content=""))
    input_data = {"messages": st.session_state.messages}

    try:
        for step in graph.stream(input_data, stream_mode = "values", config = config):
            ai_msg = [m for m in step["messages"] if isinstance(m, AIMessage)]
            if ai_msg:
                current_content = ai_msg[-1].content
                if current_content != last_ai_content:
                    full_response = current_content
                    response_placeholder.markdown(full_response)
                    last_ai_content = current_content
    
    except Exception as e:
        full_response = "I am sorry, I don't know to asnswer that"
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append(AIMessage(content=full_response))
    

def run_chatbot(graph, config):
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("💬 RAG Chatbot")

    initialize_session()
    display_chat_history()

    if handle_user_input():
        stream_response(graph, config)

