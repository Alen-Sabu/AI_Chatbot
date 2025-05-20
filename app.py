import os
from langchain.chat_models import init_chat_model

from langchain_core.messages import HumanMessage, AIMessage

from langchain_huggingface import HuggingFaceEmbeddings
import os

from langchain_chroma import Chroma

from utils import load_csv_as_documents

from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_core.prompts import ChatPromptTemplate

from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from langgraph.graph import START, StateGraph, MessagesState

from dotenv import load_dotenv

import streamlit as st 

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
# Load environment variables
load_dotenv()

# Initializing llm model
model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Initializing the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # You can change the model here
)

# Vector DB
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_model,
    persist_directory="chroma_db",  # Where to save data locally, remove if not necessary
)

# Load and prepare documents
documents = load_csv_as_documents()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True,
)
all_splits = text_splitter.split_documents(documents)
vector_store.add_documents(documents=all_splits)




@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = model.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

### Node 2: generate answer and update messages
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. Use three sentences maximum and keep the "
        "answer concise. If you are unable to answer from the the given content give answer on your own"
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = model.invoke(prompt)
    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
config = {"configurable": {"thread_id": "abc123"}}
graph = graph_builder.compile(checkpointer=memory)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("💬 RAG Chatbot")

# Initialize chat history in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past chat messages
for msg in st.session_state.messages:
    role = "🧑 You" if isinstance(msg, HumanMessage) else "🤖 Bot"
    st.chat_message(role).write(msg.content)
 # User input field
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to history
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    st.chat_message("🧑 You").write(prompt)

    # Display placeholder for streaming response
    response_container = st.chat_message("🤖 Bot")
    response_placeholder = response_container.empty()

    # Stream response from LangGraph
    full_response = ""
    input_data = {"messages": st.session_state.messages}

    last_ai_content = None  # Track last message to avoid repeated chunks

    for step in graph.stream(input_data, stream_mode="values", config=config):
        ai_msgs = [m for m in step["messages"] if isinstance(m, AIMessage)]
        if ai_msgs:
            current_content = ai_msgs[-1].content
            # Only update if new content has changed
            if current_content != last_ai_content:
                full_response = current_content
                response_placeholder.markdown(full_response + "▌")
                last_ai_content = current_content

    # Final response
    ai_msg = AIMessage(content=full_response)
    st.session_state.messages.append(ai_msg)
    response_placeholder.markdown(full_response)
