# 🤖 AI Chatbot with LangChain & Streamlit

This is an AI-powered chatbot built using LangChain, and Streamlit. It enables conversational interaction over a custom dataset with support for semantic search, streaming responses, and memory.

## 🚀 Features

* 🔍 Semantic search over CSV documents using ChromaDB
* 💬 Conversational memory using LangGraph and state-based logic
* 🔗 Tool integration via LangChain ToolNode
* ⚡ Streamed AI responses
* 🧠 In-memory or persistent vector storage
* 📄 CSV document ingestion and text splitting
* Showing previous chats

## 📁 Project Structure

```
├── app.py                  # Entry point of the application
|__ chatbot_ui.py           # Setting up the streamlit functions
├── chatbot/
│   ├── __init__.py
│   ├── embeddings.py       # Embedding model setup
|   |── llm.py              # LLM model setup
│   ├── utils.py            # Document loading, helpers
│   |── graph_setup.py      # LangGraph pipeline definition
|   |── graph_setup.py      # LangGraph  nodes setup
|   |── retrieve.py         # Retreiving the data
|   |── vectorestore.py     # Creating the vector store 
├── chroma_db/              # Folder for persistent ChromaDB storage
├── requirements.txt
└── README.md
```

## 💠 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Alen-Sabu/AI_Chatbot.git
cd AI_Chatbot
```

2. **Create virtual environment & install dependencies**

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run app.py
```

4. **Prepare your data**

Put your CSV file in the location expected by `load_csv_as_documents()` from `utils.py`.

## 🧠 Technologies Used

* [LangChain](https://github.com/langchain-ai/langchain)
* [Streamlit](https://streamlit.io/)
* [LangGraph](https://github.com/langchain-ai/langgraph)
* [ Groq](https://platform.openai.com/) for LLM access


## Other References and Technologies
* ChatGPT
* LangChain - reference for creating the rag: https://python.langchain.com/docs/tutorials/rag/

## ⚙️ Config

The chatbot is configured via a LangGraph state graph in `graph.py` with nodes:

* `query_or_respond` – Entry point to decide if tools are needed
* `tools` – Retrieval or external tool calls
* `generate` – Response generation using LLMs

## 📌 Notes

* Vectorstore is persistent via ChromaDB or can be run in-memory
* CSV documents are chunked using `RecursiveCharacterTextSplitter`
* Can be extended with more tools or integrated into a backend

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Project File Explanation
# llm.py
sets up and initializes the Large Language Model (LLM) 
to be used across the chatbot pipeline using the LangChain abstraction.

# embeddings.py
initializes a sentence embedding model from Hugging Face,
which is used to convert textual data into numerical vector representations.
These embeddings are later stored in a vector database for similarity search during retrieval

# vectorstore.py
get_vector_store()  -   is responsible for creating and
                        returning a vector database (vector store) that enables
                        semantic search (retrieving documents based on meaning, not keywords).

# utils.py
load_csv_as_documents() - To convert structured CSV data into LangChain-compatible Document objects
                          for indexing in a vector database enabling semantic retrieval 
                          during chatbot interactions.

# retriever.py
retrieve() - Retrieve information related to a query.

# graph_nodes.py
 query_or_respond() - This function decides whether the LLM needs 
                        to call an external retrieval tool to fetch context
                        from the memory

 generate() - This function handles the final answer generation.
                It takes the output of the retrieval tool (retrieved documents) and 
                constructs a prompt to generate the final, concise LLM response.

# graph_setup.py
build_graph() - It builds a graph  and It defines how your chatbot processes a message
                 — from checking whether it needs tools to generating a response.

# chatbot_ui.py
 initialize_session(): Initialize session state to store chat history

 display_chat_history(): Display previous messages from session history 

 handle_user_input(): Capture and store user input 

 stream_response(graph, config): Stream AI response and update UI. 
                                 Takes graph and config from app.py\

 run_chatbot(graph, config) : main setup of the ui that calls all the above functions and pass graph and config to stream_response

# app.py
 - takes graph, config from build_graph function and pass it on to run_chatbot function
 - calls the run_chatbot for streamlit ui


