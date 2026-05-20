# KinetiCore

KinetiCore is a Retrieval-Augmented Generation (RAG) pipeline application. It utilizes local components to ingest documents, generate vector embeddings, and retrieve context to answer queries using a Qdrant local database.

## Structure
- `app/ingestion`: Data processing including loading PDFs and chunking text.
- `app/database`: Management of vector embeddings via `qdrant-client`.
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
