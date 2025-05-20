import csv
from langchain.schema import Document


csv_path = "c:/Users/Admin/OneDrive/Desktop/PROJECTS/internship_projects/ai_chatbot/ai_chatbot/data/data.csv"

def load_csv_as_documents(csvpath = csv_path):
    docs = []

    with open(csvpath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Combine category, question, and answer into a single string
            combined_text = f"Category: {row['category']}\nQuestion: {row['question']}\nAnswer: {row['answer']}"
            docs.append(Document(page_content=combined_text, metadata={"category": row["category"]}))
    
    return docs
