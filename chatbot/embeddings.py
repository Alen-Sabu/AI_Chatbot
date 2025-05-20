from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
"""
This module initializes a sentence embedding model from Hugging Face,
 which is used to convert textual data into numerical vector representations.
These embeddings are later stored in a vector database for similarity search during retrieval.
"""
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
