import os
from langchain.chat_models import init_chat_model

from langchain_core.messages import HumanMessage

from langchain.embeddings import HuggingFaceBgeEmbeddings
import os

from langchain_chroma import Chroma

from utils import load_csv_as_documents

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from langgraph.graph import START, StateGraph

from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Initializing llm model
model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Initializing the embedding model
embedding_model = HuggingFaceBgeEmbeddings(
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
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
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

result = graph.invoke({"question": "What is Task Decomposition?"})

print(f'Answer: {result["answer"]}')