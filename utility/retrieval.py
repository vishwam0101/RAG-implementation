import chromadb
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Set up the ChromaDB persistent client
client = chromadb.PersistentClient(path="./vector_store")

# Retrieve or create the collection
collection_name = "ragdb"
collection = client.get_or_create_collection(name=collection_name)

def retrieve(query, top_k=3):
    """
    Retrieves the top_k most relevant document chunks from the vector database for the given query.
    """
    # Generate the embedding for the query
    query_embedding = embedder.encode(query).tolist()

    # Perform the query on the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents']
    )

    # Extract the documents from the results
    documents = results.get('documents', [[]])[0]

    # Return the documents if found, else return a default message
    return documents if documents else ["No relevant context found."]
