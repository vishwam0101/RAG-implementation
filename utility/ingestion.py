import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import os

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB PERSISTENT client (CORRECT!)
client = chromadb.PersistentClient(path="./vector_store")

# Load or create collection
collection_name = "ragdb"
if collection_name not in [col.name for col in client.list_collections()]:
    collection = client.create_collection(name=collection_name)
else:
    collection = client.get_collection(name=collection_name)


def ingest_pdf(pdf_path):
    # Read the PDF
    reader = PdfReader(pdf_path)
    all_text = ""
    import re

    for page in reader.pages:
        raw_text = page.extract_text()
        if raw_text:
            clean_text = re.sub(r"\[\d+\]", "", raw_text)        # remove [23], [1], etc.
            clean_text = re.sub(r"\s+", " ", clean_text).strip() # normalize whitespace
            clean_text = re.sub(r"[^\x00-\x7F]+", " ", clean_text) # remove non-ASCII
            all_text += clean_text + "\n"


    # Chunk the text
    chunk_size = 1000  # characters
    chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]

    # Add each chunk to the vector database
    for idx, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(idx)]
        )

    print(f"Ingestion complete: {len(chunks)} chunks stored and saved to disk!")

# Quick script to call ingestion
if __name__ == "__main__":
    pdf_path = os.path.join(os.path.dirname(__file__), "AgenticRAG.pdf")
    ingest_pdf(pdf_path)
