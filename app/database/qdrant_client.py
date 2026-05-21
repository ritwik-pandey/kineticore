import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


client = QdrantClient(path="./qdrant_local_db")
collection_name = "my_documents"
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def qdrant(chunks, vector_embeddings):
    

    if not client.collection_exists(collection_name=collection_name):
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, vector_embeddings)):
        point_id = str(uuid.uuid4()) 
    
        payload = {
            "text": chunk.page_content, 
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page", 0)
        }
        
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    print(f"Upserting {len(points)} vectors into Qdrant...")
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print("Vectors stored successfully!\n")

   

def retrieve_documents(query: str, top_k: int = 3):
    print(f"\n🔍 Searching for: '{query}'")
    
    
    query_vector = embeddings_model.embed_query(query)
    
    
    search_response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )
    
    
    retrieved_chunks = []
    for hit in search_response.points:
        retrieved_chunks.append({
            "score": hit.score,
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "page": hit.payload.get("page", "N/A")
        })
        
    return retrieved_chunks

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

    for rank, hit in enumerate(dense_results.points):
        text = hit.payload["text"]
        if text not in fused_scores:
            fused_scores[text] = {"score": 0.0, "source": hit.payload.get("source", "Unknown")}
        fused_scores[text]["score"] += 1.0 / (k + rank + 1)

    for rank, doc in enumerate(sparse_results):
        text = doc.page_content
        if text not in fused_scores:
            fused_scores[text] = {"score": 0.0, "source": doc.metadata.get("source", "Unknown")}
        fused_scores[text]["score"] += 1.0 / (k + rank + 1)

    
    ranked_results = sorted(fused_scores.items(), key=lambda item: item[1]["score"], reverse=True)

    final_output = []
    for text, data in ranked_results[:top_k]:
        final_output.append({
            "text": text,
            "source": data["source"],
            "hybrid_score": data["score"]
        })
    output = rerank_documents(query, final_output,3)
    return output

def rerank_documents(query: str, retrieved_chunks: list, top_k: int = 3):
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"\n🧠 Re-ranking {len(retrieved_chunks)} candidates...")
    
    if not retrieved_chunks:
        return []

    
    sentence_pairs = []
    for chunk in retrieved_chunks:
        sentence_pairs.append([query, chunk["text"]])
        
   
    scores = reranker.predict(sentence_pairs)
    
    for i, chunk in enumerate(retrieved_chunks):
        chunk["rerank_score"] = float(scores[i])
        
    ranked_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return ranked_chunks[:top_k]

