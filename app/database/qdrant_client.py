import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings

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

