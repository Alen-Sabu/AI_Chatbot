import csv
from langchain.schema import Document


csv_path = "c:/Users/Admin/OneDrive/Desktop/PROJECTS/internship_projects/ai_chatbot/ai_chatbot/data/data.csv"

def load_csv_as_documents(csvpath = csv_path):
    docs = []

    with open(csvpath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Combin title, header, content into one string per document
            combined_text = f"{row['post_title']}\n{row['post_header']}\n{row['post_content']}"
            docs.append(Document(page_content=combined_text))
        return docs
