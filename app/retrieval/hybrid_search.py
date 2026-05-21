from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

client = QdrantClient(path="./qdrant_local_db")
collection_name = "my_documents"
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_all_chunks_from_qdrant():
    """Retrieve all stored chunks from Qdrant to populate BM25."""
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    
    chunks = []
    for point in points:
        text = point.payload.get("text", "")
        source = point.payload.get("source", "unknown")
        chunks.append(Document(page_content=text, metadata={"source": source}))
    
    return chunks

def hybrid(query: str, top_k: int = 3):
    print(f"performing hybrid search: '{query}'")

    # Fetch chunks from the database
    chunks = get_all_chunks_from_qdrant()
    if not chunks:
        print("No documents found in database. Run ingestion first.")
        return []

    query_vector = embeddings_model.embed_query(query)
    
    dense_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    sparse_results = bm25_retriever.invoke(query)
    sparse_results = sparse_results[:5]
    fused_scores = {}

    k=60

    # Process Dense Ranks
    for rank, hit in enumerate(dense_results.points):
        text = hit.payload["text"]
        if text not in fused_scores:
            fused_scores[text] = {"score": 0.0, "source": hit.payload.get("source", "Unknown")}
        # Calculate RRF for Dense
        fused_scores[text]["score"] += 1.0 / (k + rank + 1)

    # Process Sparse Ranks
    for rank, doc in enumerate(sparse_results):
        text = doc.page_content
        if text not in fused_scores:
            fused_scores[text] = {"score": 0.0, "source": doc.metadata.get("source", "Unknown")}
        # Calculate RRF for Sparse
        fused_scores[text]["score"] += 1.0 / (k + rank + 1)

    # ==========================================
    # 4. FINAL SORTING
    # ==========================================
    # Sort the dictionary by the highest fused score
    ranked_results = sorted(fused_scores.items(), key=lambda item: item[1]["score"], reverse=True)

    # Format and return the absolute top_k results
    final_output = []
    for text, data in ranked_results[:top_k]:
        final_output.append({
            "text": text,
            "source": data["source"],
            "hybrid_score": data["score"]
        })
    print(final_output)
    
