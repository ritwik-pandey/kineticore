from ingestion.loader import process_document
from ingestion.chunker import chunk_data
from ingestion.embeddings import vectorEmbeddings
from database.qdrant_client import qdrant
from database.qdrant_client import retrieve_documents
from generation.generator import answer_question

file_path = "./pdf/sample.pdf"
documents = process_document(file_path)
chunks = chunk_data(documents)
embeddings = vectorEmbeddings(chunks)
qdrant(chunks,embeddings)

query="Write the code of bellman ford"
results = retrieve_documents(query, top_k=2)
print(answer_question(results,query))

