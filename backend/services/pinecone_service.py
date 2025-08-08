from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from core.config import key

# 1. Initialize Pinecone client
pc = Pinecone(api_key=key.pinecone_api_key)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
# Delete old index
pc.delete_index(key.pinecone_index)

# Create new index with correct dimension
pc.create_index(
    name=key.pinecone_index,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# 2. Correctly check if the index exists
# Use 'idx.name' and iterate over 'pc.list_indexes().indexes'
index_name = key.pinecone_index
if index_name not in [idx.name for idx in pc.list_indexes().indexes]:
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 output dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# 3. Correctly connect to the index from the client instance 'pc'
index = pc.Index(index_name)

def embed_and_store_chunks(chunks: list[str], source_id: str):
    """Embed and store text chunks in Pinecone."""
    vectors = model.encode(chunks).tolist()
    payload = [
        (
            f"{source_id}_{i}",  # unique ID
            vec,
            {"text": chunk}      # metadata
        )
        for i, (vec, chunk) in enumerate(zip(vectors, chunks))
    ]
    index.upsert(vectors=payload)

def search_chunks(query: str, k: int = 5) -> list[str]:
    """Search Pinecone for top-k similar chunks."""
    query_vector = model.encode(query).tolist() # Note: model expects a list of texts
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]