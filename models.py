"""
Single source of truth of all LLM and embedding models.
Uses the official langchain-ollama package for stable local inference.
"""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    LLM_TEMPERATURE
)

def get_llm(temperature: float = None) -> ChatOllama:
    """
    Returns the primary Chat LLM for the agent.

    Args:
        temperature: Override default temperature (None uses config default)

    Returns:
        ChatOllama instance for local inference
    """
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    return ChatOllama(
        model=LLM_MODEL,
        temperature=temp,
        base_url = "http://localhost:11434"
    )

def get_embeddings() -> OllamaEmbeddings:
    """
    Returns the embedding model for vectorizing documents.
    
    Returns:
        OllamaEmbeddings instance for local embedding generation
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434"   # Ollama embeddings endpoint
    )

#  --- UTILITY FUNCTIONS FOR TESTING ---

def test_llm_connection():
    """Test if the LLM is accessible and responding."""
    try:    
        llm = get_llm()
        response = llm.invoke("Say 'Connection Successful' in exactly 3 words.")
        print(f'LLM Test: {response.content.strip()}')
        return True
    except Exception as e:
        print(f'LLM Connection Failed: {e}')
        return False

def test_embeddings_connection():
    """Test if the embedding model is accessible."""
    try:  
        embeddings = get_embeddings()
        vector = embeddings.embed_query("Test embedding")
        print(f'Embeddings Test: Generated vector - {vector[:3]}... of length {len(vector)}')
        return True
    except Exception as e:
        print(f'Embeddings Connection Failed: {e}')
        return False

# --- MAIN EXECUTION (for testing) ---
    
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Model Initialization")
    print("=" * 50)

    # Test LLM
    print("\n[1/2] Testing LLM connection...")
    llm_ok = test_llm_connection()

    # Test Embeddings
    print("\n[2/2] Testing Embeddings connection...")
    embeddings_ok = test_embeddings_connection()

    # Summary
    print("\n" + "=" * 50)
    if llm_ok and embeddings_ok:
        print("All models initialized successfully!")
    else:
        print("Some models failed. Check Ollama is running: `ollama serve`")
    print("=" * 50)
