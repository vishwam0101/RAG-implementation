# utility/test_retrieval.py

import chromadb
from sentence_transformers import SentenceTransformer

# Load persistent ChromaDB client (CORRECT way)
client = chromadb.PersistentClient(path="./vector_store")

# Load the collection
collection = client.get_collection(name="ragdb")

# Initialize the same embedding model used during ingestion
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def query_rag(query_text, top_k=3):
    query_embedding = embedder.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "distances"]
    )

    return results

if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    retrieved = query_rag(user_query)

    print("\n🔍 Top matches:")
    for idx, doc in enumerate(retrieved["documents"][0]):
        print(f"\nResult {idx+1}:")
        print(doc)
        print(f"Distance: {retrieved['distances'][0][idx]:.4f}")
