from langchain_chroma import Chroma
from chatbot.embeddings import embedding_model
from chatbot.utils import load_csv_as_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_vector_store():
    """
    The function get_vector_store() is responsible for creating and
    returning a vector database (vector store) that enables
    semantic search (retrieving documents based on meaning, not keywords).
    """
    documents = load_csv_as_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(documents)

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding_model,
        persist_directory="chroma_db",
    )

    vector_store.add_documents(documents=all_splits)
    return vector_store
