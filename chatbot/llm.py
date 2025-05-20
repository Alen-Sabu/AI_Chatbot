from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

"""
This code sets up and initializes the Large Language Model (LLM) 
to be used across the chatbot pipeline using the LangChain abstraction.
"""

load_dotenv()
model = init_chat_model("llama3-8b-8192", model_provider="groq")
