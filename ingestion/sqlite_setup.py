import pandas as pd
import sqlite3
from pathlib import Path
from config import DATA_DIR, SQLITE_DB_PATH

def setup_sqlite_database():
    """Reads the Kaggle CSV and normalizes it into 3 relational SQLite tables."""
    csv_path = DATA_DIR / "bank_transactions.csv" 

    if not csv_path.exists():
        print(f'Error: CSV file not found at {csv_path}')
        return False

    print(f'Loading data from {csv_path}...')
    df = pd.read_csv(csv_path)

    # 1. Basic cleanup: lowercase and replace spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 2. EXPLICIT MAPPING: Ensure column names match exactly what we tell the LLM
    column_mapping = {
        'transactionid': 'transaction_id',
        'accountid': 'account_id',
        'transactionamount': 'transaction_amount',
        'transactiondate': 'transaction_date',
        'transactiontype': 'transaction_type',
        'transactionduration': 'transaction_duration',
        'accountbalance': 'current_balance',
        'channel': 'primary_channel',
        'location': 'location',
        'deviceid': 'device_id',
        'ip_address': 'ip_address',
        'merchantid': 'merchant_id',
        'customerage': 'customer_age',
        'customeroccupation': 'customer_occupation',
        'loginattempts': 'login_attempts'
 }
    df.rename(columns=column_mapping, inplace=True)
    
    print("Normalizing data into 3 relational tables...")

    # 1. Customers Table
    customers_df = df[['account_id', 'customer_age', 'customer_occupation']].drop_duplicates(subset=['account_id'])

    # 2. Accounts Table (Mocking a current balance and primary channel for demo purposes)
    # We'll group by account to get the most recent balance and most frequent channel
    accounts_df = df[['account_id', 'current_balance', 'primary_channel']].drop_duplicates(subset=['account_id'])

    # 3. Transactions Table
    transactions_df = df[['transaction_id', 'account_id', 'transaction_amount', 'transaction_date', 'transaction_type', 'location', 'device_id', 'ip_address', 'merchant_id', 'login_attempts']].copy()

    print("Writing to SQLite database...")
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            customers_df.to_sql('customers', conn, if_exists='replace', index=False)
            accounts_df.to_sql('accounts', conn, if_exists='replace', index=False)
            transactions_df.to_sql('transactions', conn, if_exists='replace', index=False)

            # Create indexes for faster SQL Agent queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_account ON transactions(account_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_date ON transactions(transaction_date)")

        print(f'Success! Database created at: {SQLITE_DB_PATH}')
        print(f"Tables created: 'customers', 'accounts', 'transactions'")
        return True
    except Exception as e:
        print(f'Failed to create SQLite databse: {e}')
        return False

if __name__ == "__main__":
    print("="* 50)
    print("Structured Data Ingestion Pipeline (SQLite)")
    print("="*50)
    setup_sqlite_database()