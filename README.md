A typical RAG application has two main components:

Indexing: a pipeline for ingesting data from a source and indexing it. 

Retrieving and generation: the actual RAG chain, which takes the user query at run time and retrieves data from the index, then passes to the model.

Indexing:
- First we need to load our data. This is done with Document Loaders
- Split: Text splitters break large Documents into smaller chunks. This is useful both for indexing data and passing it into a model, as large chunks are harder to search over and won't fit in a model's finite context window
- Store: We need somewhere to store and index our splits, so that they can be searched over later. This is done using a VectorStore and Embeddings model

Retrieval and generation:
- Given a user input, relevant splits are retrieved form storage using Retriever
- Generate: A chat model produces answer using a prompt that includes both the question with the retrieved data.

- Creating a model using Grog

- Creating a embedding using Hugging Face

- Creating a vector store using Chroma

- Document Loading

- Splitting documents

- We'll use LangGraph to tie together the retrieval and generation steps into a single application. This will bring a number of benefits:

    * We can define our application logic once and automatically support multiple invocation modes, including streaming, async, and batched calls.
    * We get streamlined deployments via LangGraph Platform.
    * LangSmith will automatically trace the steps of our application together.
    * We can easily add key features to our application, including persistence and human-in-the-loop approval, with minimal code changes.
 To use LangGraph, we need to define three things:

    * The state of our application;
    * The nodes of our application (i.e., application steps);
    * The "control flow" of our application (e.g., the ordering of the steps).

- Creating State of the application