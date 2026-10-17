import sqlite3
import pandas as pd
from langchain_core.tools import tool
from langchain_chroma import Chroma
from models import get_embeddings
from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

# --- 1. SQL EXECUTION TOOL ---
@tool
def execute_sql_query(sql_query: str) -> str:
    """
    Execute a read-only SQLite query on the bank transactions database.
    Use this tool when the user asks for specific numbers, counts, averages, 
    or transaction details (e.g., "How many transactions over $5000?").
    
    Args:
        sql_query: A valid, read-only SQLite SELECT query.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(SQLITE_DB_PATH)

        # Execute the query using pandas for easy formatting
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        # Format the dataframe as a clean string for the LLM to read
        if df.empty:
            return "Query executed successfully, but returned no results."

        return df.to_markdown(index=False)
    
    except Exception as e:
        # Return the error to the LLM so it can self-correct!
        return f"SQL Execution Error: {str(e)}. Please check your syntax and table names."

# --- 2. RAG RETRIEVAL TOOL --- 
@tool
def retrieve_policy_context(query: str) -> str:
    """
    Search the vector database for bank fraud policies, rules, and procedures.
    Use this tool when the user asks about rules, policies, SOPs, or definitions 
    (e.g., "What is the policy for late fees?" or "What are the Red Flag Indicators?").
    
    Args:
        query: A natural language search query.
    """
    try:
        # Initialize the vector store and retriever
        embeddings = get_embeddings()
        vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )

        # Retrieve top 3 most relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        # Format the retrieved documents
        if not docs:
            return "No relevant policy documents found."

        context = "\n\n---\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
            for doc in docs
        ])
        return context
    
    except Exception as e:
        return f"Execution Error: {str(e)}. Please"

# List of tools to give to the LLM
AGENT_TOOLS = [execute_sql_query, retrieve_policy_context]