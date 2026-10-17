"""
Master Data Ingestion Pipeline.
Orchestrates the loading, processing, and storage of both structured (CSV) 
and unstructured (PDF) data into their respective databases.
"""
import sys
from pathlib import Path

# Ensure we can import from the root directory
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, POLICIES_DIR
from ingestion.sqlite_setup import setup_sqlite_database
from ingestion.vector_store import setup_vector_store

def run_full_ingestion():
    """
    Master function to run the entire data ingestion pipeline.
    Returns True if both structured and unstructured data are loaded successfully.
    """
    print("=" * 60)
    print("Starting Full Data Ingestion Pipeline")
    print("=" * 60)

    csv_path = DATA_DIR / "bank_transactions.csv"
    if not csv_path.exists():
        print(f"Error: Cannot find {csv_path}. Please add your CSV to the data/ folder.")
        return False

    if not POLICIES_DIR.exists() or not any(POLICIES_DIR.glob("*.pdf")):
        print(f"Error: Cannot find PDF files in {POLICIES_DIR}. Please add your policy PDFs.")
        return False

    print("Source data verified. Beginning processing...\n")

    # ---------------------------------------------------------
    # STEP 1: Structured Data (SQLite)
    # ---------------------------------------------------------
    print("[1/2] Processing Structured Data (Bank Transactions)...")
    print("-" * 60)
    sqlite_sucess = setup_sqlite_database()

    if not sqlite_sucess:
        print("Structured data ingestion failed. Aborting pipeline.")
        return False

    print("Structured data ingestion complete.\n")

    # ---------------------------------------------------------
    # STEP 2: Unstructured Data (Vector Store)
    # ---------------------------------------------------------
    print("[2/2] Processing Unstructured Data (Fraud Policies)...")
    print("-" * 60)
    vector_sucess = setup_vector_store()

    if not vector_sucess:
        print("Unstructured data ingestion failed.")
        return False
        
    print("Unstructured data ingestion complete.\n")

    # ---------------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------------
    print("=" * 60)
    print("DATA INGESTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Your agent is now ready to query:")
    print(f"  • SQLite Database: {DATA_DIR / 'finance_bank.db'}")
    print(f"  • Chroma Vector DB:  {DATA_DIR / 'chroma_db'}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    sucess = run_full_ingestion()
    sys.exit(0 if sucess else 1)