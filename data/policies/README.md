# Policy Documents Directory

Place your Anti-Fraud and Compliance PDF documents here.

## Required Documents
- For this demo, I used `Anti_Fraud_Policy_2026.pdf`, `Anti-Fraud-Policy-April-2023.pdf`, `fraud-rm-policy.pdf`.
- You can also add any additional compliance or regulatory PDFs.

## How It Works
The ingestion pipeline (`ingestion/vector_store.py`) will automatically:
1. Load all PDF files from this directory
2. Chunk them into 800-token segments with 100-token overlap
3. Embed them using `nomic-embed-text` via Ollama
4. Store them in ChromaDB for RAG retrieval

## Note
These files are excluded from Git via `.gitignore` to keep the repository lightweight and protect sensitive internal documents.