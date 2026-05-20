from langchain_huggingface import HuggingFaceEmbeddings

def vectorEmbeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_chunks = [chunk.page_content for chunk in chunks]
    print(f"Generating embeddings for {len(text_chunks)} chunks... This may take a moment.")
    vector_embeddings = embeddings.embed_documents(text_chunks)

    return vector_embeddings