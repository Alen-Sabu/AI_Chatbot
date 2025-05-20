import os
from langchain.chat_models import init_chat_model

from langchain_core.messages import HumanMessage

from langchain_huggingface import HuggingFaceEmbeddings
import os

from langchain_chroma import Chroma

from utils import load_csv_as_documents

from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_core.prompts import ChatPromptTemplate

from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from langgraph.graph import START, StateGraph

from dotenv import load_dotenv

import streamlit as st 

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

# Prompt and LangGraph pipeline
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the following context to answer the question.
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k = 1)
    return {"context": retrieved_docs}

def generate(state: State):

    if not state["context"]:
        return {"answer": "Sorry, I couldn't find relevant information in the provided document."}

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(messages)
    return {"answer": response.content}

graph = (
    StateGraph(State)
    .add_sequence([retrieve, generate])
    .add_edge(START, "retrieve")
    .compile()
)
st.title("RAG Chatbot")



query = st.text_input("Ask your question:")
if query:

    result = graph.invoke({"question": query})
    st.markdown("### Retrieved Context:")
    for doc in result["context"]:
        st.write(doc.page_content)

    st.markdown("### Answer:")
    docs_content = "\n\n".join(doc.page_content for doc in result["context"])
    messages = prompt.invoke({"question": query, "context": docs_content})

    # Stream the response
    response_placeholder = st.empty()
    streamed_text = ""

    for chunk in model.stream(messages):
        streamed_text += chunk.content
        response_placeholder.markdown(streamed_text + "▌")

    response_placeholder.markdown(streamed_text) 

