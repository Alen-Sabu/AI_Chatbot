rag_chatbot/
│
├── chatbot/
│   ├── __init__.py
│   ├── llm.py                    # LLM model init (Groq)
│   ├── embeddings.py             # Embedding model init
│   ├── vectorstore.py            # Chroma DB setup
│   ├── utils.py                  # CSV loader + converter to LangChain docs
│   ├── retriever.py              # Retrieval tool definition
│   ├── graph_nodes.py            # LangGraph nodes: query/respond, generate
│   ├── graph_setup.py            # LangGraph builder and config
│
├── app.py                        # main program execution
|__ chatbotui.py                  # Streamlit frontend logic
│
|
│
├── data/                         # Your raw CSV or dataset
│   └── my_dataset.csv
│
├── .env                          # Environment variables (e.g., API keys)
├── requirements.txt              # Python packages
└── README.md                     # Project overview


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


How to run the project ?
 - Install dependencies
    Ensure you have a virtual environment activated, then install requirements:
    pip install -r requirements.txt

 - Create a .env file and mention your hugging face token and grog api key

 -  Run the chatbot with Streamlit using the command: streamlit run app.py


