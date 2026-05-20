from ingestion.loader import process_document
from ingestion.chunker import chunk_data
from ingestion.embeddings import vectorEmbeddings
from database.qdrant_client import qdrant
from database.qdrant_client import retrieve_documents
from generation.generator import answer_question
from monitoring.file_watcher import start_watcher

observer = start_watcher()

try:
    while True:
        print("\nEnter the query (or 'quit' to exit): ")
        query = input("You: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        results = retrieve_documents(query, top_k=2)
        print(answer_question(results, query))
except KeyboardInterrupt:
    pass
finally:
    print("\nShutting down...")
    observer.stop()
    observer.join()
