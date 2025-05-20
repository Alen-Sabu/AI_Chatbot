import csv
from langchain.schema import Document


csv_path = "data/data.csv"

def load_csv_as_documents(csvpath = csv_path):
    """
    To convert structured CSV data into LangChain-compatible Document objects
    for indexing in a vector database enabling semantic retrieval 
    during chatbot interactions.
    """
    docs = []

    with open(csvpath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Combine category, question, and answer into a single string
            combined_text = f"Category: {row['category']}\nQuestion: {row['question']}\nAnswer: {row['answer']}"
            docs.append(Document(page_content=combined_text, metadata={"category": row["category"]}))
    
    return docs
