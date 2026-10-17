import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from models import get_embeddings
from config import POLICIES_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

def setup_vector_store():
    """Loads PDFs, chunks them, embeds them, and stores them in ChromaDB."""

    if not os.path.exists(POLICIES_DIR):
        print(f'Error: Policies directory not found at {POLICIES_DIR}')
        return False

    print(f'Loading PDFs from {POLICIES_DIR}...')
    loader = PyPDFDirectoryLoader(str(POLICIES_DIR))
    documents = loader.load()
    print(f'Loaded {len(documents)} document pages.')

    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ". ", " ", ""]  # Smart splitting for policies
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Created {len(chunks)} text chunks.')

    print("Embedding and storing in ChromaDB...")
    try:
        # Initialize embeddings from our models.py
        embeddings = get_embeddings()

        # Create or load persistent Chroma database
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
            collection_name=CHROMA_COLLECTION_NAME
        )
        print(f'Success! Vector store saved to: {CHROMA_PERSIST_DIR}')
        print(f"Collection: '{CHROMA_COLLECTION_NAME}'")

        # Quick test retrieval
        print('\n Testing retrieval...')
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        test_query = "What are the red flag indicators for fraud?"
        docs = retriever.invoke(test_query)
        print(f"Top result snippet: '{docs[0].page_content[:100]}...'")

        return True
    except Exception as e:
        print(f'Failed to create vector store: {e}')
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Unstructured Data Ingestion (ChromaDB)")
    print("=" * 50)
    setup_vector_store()
    