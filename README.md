rag_chatbot/
│
├── chatbot/
│   ├── __init__.py
│   ├── llm.py                    # LLM model init (Groq)
│   ├── embeddings.py             # Embedding model init
│   ├── vectorstore.py            # Chroma DB setup
│   ├── utils.py            # CSV loader + converter to LangChain docs
│   ├── retriever.py              # Retrieval tool definition
│   ├── graph_nodes.py            # LangGraph nodes: query/respond, generate
│   ├── graph_setup.py            # LangGraph builder and config
│
├── ui/
│   └── app.py                    # Streamlit frontend logic
│
├── chroma_db/                    # Vector store persistence directory
│
├── data/                         # Your raw CSV or dataset
│   └── my_dataset.csv
│
├── .env                          # Environment variables (e.g., API keys)
├── requirements.txt              # Python packages
└── README.md                     # Project overview
