# KinetiCore

KinetiCore is a Retrieval-Augmented Generation (RAG) pipeline application. It utilizes local components to automatically ingest documents via folder monitoring, generate vector embeddings, perform hybrid search with reranking, and retrieve context to answer queries using a Qdrant local database.

## Structure
- `app/ingestion`: Data processing including loading PDFs and chunking text.
- `app/database`: Management of vector embeddings via `qdrant-client`.
- `app/monitoring`: Automatic file ingestion and folder monitoring.
- `app/retrieval`: Hybrid search combining dense and sparse retrieval, along with a reranking pipeline.
- `app/generation`: Local generation component for answering questions.
- `requirements.txt`: Python dependencies.

## Setup

1. Create a virtual environment and install requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables in `.env`.

3. Run the application logic from `app/main.py`:
```bash
cd app
python main.py
```
