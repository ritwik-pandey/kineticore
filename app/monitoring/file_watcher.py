import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingestion.loader import process_document
from ingestion.chunker import chunk_data
from ingestion.embeddings import vectorEmbeddings
from database.qdrant_client import qdrant
from database.qdrant_client import retrieve_documents
from generation.generator import answer_question

WATCH_DIRECTORY = "./pdf"

def process_and_store(file_path):
    print(f"   -> Loading {file_path}...")
    documents = process_document(file_path)
    chunks = chunk_data(documents)
    embeddings = vectorEmbeddings(chunks)
    qdrant(chunks,embeddings)
    print(f"   -> [SUCCESS] Indexed into Qdrant!")

class DocumentIngestionHandler(FileSystemEventHandler):
    
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        valid_extensions = ['.txt', '.pdf', '.md']
        _, extension = os.path.splitext(file_path)
        
        if extension.lower() in valid_extensions:
            print(f"\n📄 New file detected: {os.path.basename(file_path)}")
            print("🚀 Triggering automatic ingestion pipeline...")
            
            try:
              
                time.sleep(1) 
                process_and_store(file_path)
            except Exception as e:
                print(f"   -> [ERROR] Failed to process file: {e}")
        else:
            print(f"\n⚠️ Ignored unsupported file type: {os.path.basename(file_path)}")

def start_watcher():
    if not os.path.exists(WATCH_DIRECTORY):
        os.mkdir(WATCH_DIRECTORY)
    
    event_handler = DocumentIngestionHandler()
    observer = Observer()

    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    observer.start()

    print(f"👀 Watching directory '{WATCH_DIRECTORY}' for new files...")
    print("Press Ctrl+C to stop.")

    return observer