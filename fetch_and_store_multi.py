import yfinance as yf
import pandas as pd
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta

# -------------------------
# Database Connection
# -------------------------
def get_db_connection():
    """Establish connection to MySQL database."""
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  # change if needed
            password="div@MySQL25",  # change if needed
            database="finance_data"
        )
        if connection.is_connected():
            print("✅ Connected to MySQL database")
            return connection
    except Error as e:
        print(f"❌ Error: {e}")
        return None

# -------------------------
# Get Last Date in DB
# -------------------------
def get_last_date(symbol):
    conn = get_db_connection()
    if conn is None:
        return None
    cursor = conn.cursor()
    query = "SELECT MAX(date) FROM stocks WHERE symbol = %s"
    cursor.execute(query, (symbol,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result and result[0] else None

def clean_column_names(df, symbol):
    """
    Clean and standardize column names from yfinance data.
    Properly handles MultiIndex columns from yfinance.
    """
    # Print original columns for debugging
    print(f"🔍 Original columns for {symbol}: {list(df.columns)}")
    
    # Handle MultiIndex columns (which is what yfinance typically returns)
    if isinstance(df.columns, pd.MultiIndex):
        print(f"📋 Detected MultiIndex columns - flattening...")
        
        # For MultiIndex, we want to take the second level (the actual data type)
        # since the first level is the ticker symbol
        new_columns = []
        for col_tuple in df.columns:
            if len(col_tuple) == 2:
                level0, level1 = col_tuple
                # If level1 is empty (like for Date), use level0
                # Otherwise, use level1 (the data type like 'Open', 'Close', etc.)
                if level1 == '' or pd.isna(level1):
                    new_columns.append(str(level0))
                else:
                    new_columns.append(str(level1))
            else:
                # Fallback: join all levels
                new_columns.append(' '.join(map(str, col_tuple)).strip())
        
        df.columns = new_columns
        
    else:
        # Handle regular columns (convert to string and strip whitespace)
        df.columns = [str(col).strip() for col in df.columns]
    
    print(f"✅ Cleaned columns for {symbol}: {list(df.columns)}")
    
    # Verify we have all expected columns
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_data_cols = [col for col in df.columns if col in expected_cols]
    print(f"📊 Found {len(available_data_cols)} out of {len(expected_cols)} expected data columns")
    
    return df

def get_table_schema():
    """
    Get the actual column names from the stocks table in the database.
    This helps us adapt to whatever schema already exists.
    """
    conn = get_db_connection()
    if conn is None:
        return None
    
    cursor = conn.cursor()
    try:
        # Get the table structure
        cursor.execute("DESCRIBE stocks")
        columns = cursor.fetchall()
        
        # Extract just the column names (first element of each row)
        column_names = [col[0] for col in columns]
        print(f"🏗️ Database table 'stocks' has columns: {column_names}")
        
        cursor.close()
        conn.close()
        return column_names
        
    except Exception as e:
        print(f"❌ Error getting table schema: {e}")
        cursor.close()
        conn.close()
        return None

def upsert_rows(symbol, df):
    """
    Ultra-fast bulk insert or update rows in the MySQL 'stocks' table for a given symbol.
    Adapts to the actual database schema to avoid column name mismatches.
    """
    if df.empty:
        print(f"⚠ No data to insert for {symbol}")
        return 0

    # Ensure 'Date' is a column (reset index if needed)
    if df.index.name == "Date" or df.index.name == "date":
        df = df.reset_index()
    
    # Clean and standardize column names
    df = clean_column_names(df, symbol)

    # Get the actual database schema to match our insert statement
    db_columns = get_table_schema()
    if db_columns is None:
        print(f"❌ Could not retrieve database schema for {symbol}")
        return 0

    # Check if all required columns are present in our dataframe
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    
    # Handle the Date column - it might be called 'Date' or be the index
    if "Date" not in df.columns:
        if df.index.name in ["Date", "date"] or df.index.dtype.kind == 'M':  # 'M' for datetime
            df = df.reset_index()
            df.columns = ["Date"] + list(df.columns[1:])
        else:
            print(f"❌ No Date column found for {symbol}")
            return 0
    
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns for {symbol}: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return 0

    # Convert to list of tuples for bulk insert
    try:
        records = []
        for _, row in df.iterrows():
            # Ensure date is in proper format
            date_val = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
            
            # Convert numeric values and handle any NaN values
            record = (
                symbol,
                date_val,
                float(row["Open"]) if pd.notna(row["Open"]) else 0.0,
                float(row["High"]) if pd.notna(row["High"]) else 0.0,
                float(row["Low"]) if pd.notna(row["Low"]) else 0.0,
                float(row["Close"]) if pd.notna(row["Close"]) else 0.0,
                float(row["Adj Close"]) if pd.notna(row["Adj Close"]) else 0.0,
                int(row["Volume"]) if pd.notna(row["Volume"]) else 0
            )
            records.append(record)
            
    except Exception as e:
        print(f"❌ Error preparing records for {symbol}: {e}")
        return 0

    if not records:
        print(f"⚠ No valid rows to insert for {symbol}")
        return 0

    # Database connection and bulk insert
    conn = get_db_connection()
    if conn is None:
        return 0
    cursor = conn.cursor()

    # Build the SQL dynamically based on what columns actually exist
    # This approach makes our code more robust and adaptable
    
    # Map our data columns to database columns (handling common variations)
    column_mapping = {
        'symbol': 'symbol',
        'date': 'date', 
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }
    
    # Handle the adjusted close column - check what name exists in DB
    adj_close_variations = ['adj_close', 'adjusted_close', 'adjclose', 'adj close']
    db_adj_close_col = None
    for variation in adj_close_variations:
        if variation in db_columns:
            db_adj_close_col = variation
            break
    
    if db_adj_close_col:
        column_mapping['adj_close'] = db_adj_close_col
        print(f"📋 Using '{db_adj_close_col}' as adjusted close column in database")
    else:
        print(f"⚠️ No adjusted close column found in database. Available columns: {db_columns}")
        # We'll continue without the adj_close column
        
    # Build the INSERT statement based on available columns
    if db_adj_close_col:
        sql = f"""
        INSERT INTO stocks (symbol, date, open, high, low, close, {db_adj_close_col}, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open=VALUES(open),
            high=VALUES(high),
            low=VALUES(low),
            close=VALUES(close),
            {db_adj_close_col}=VALUES({db_adj_close_col}),
            volume=VALUES(volume)
        """
    else:
        # Fallback: insert without adj_close column
        sql = """
        INSERT INTO stocks (symbol, date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open=VALUES(open),
            high=VALUES(high),
            low=VALUES(low),
            close=VALUES(close),
            volume=VALUES(volume)
        """
        # Remove adj_close from our records
        records = [(r[0], r[1], r[2], r[3], r[4], r[5], r[7]) for r in records]

    try:
        cursor.executemany(sql, records)
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"💾 Successfully executed SQL with {len(records)} records")
    except Exception as e:
        conn.rollback()
        print(f"❌ Bulk insert failed for {symbol}: {e}")
        print(f"🔍 SQL attempted: {sql[:200]}...")  # Show first 200 chars of SQL for debugging
        cursor.close()
        conn.close()
        return 0

    cursor.close()
    conn.close()
    return inserted_count

# -------------------------
# Fetch & Store Data
# -------------------------
def fetch_and_store(symbol):
    """
    Fetch stock data for a symbol and store it in the database.
    Only fetches new data since the last stored date.
    """
    last_date = get_last_date(symbol)
    if last_date:
        print(f"📅 Last date in DB for {symbol}: {last_date}")
        start_date = last_date + timedelta(days=1)
        print(f"📅 Fetching data from: {start_date.strftime('%Y-%m-%d')}")
    else:
        print(f"🆕 No existing data for {symbol}. Fetching from 2020-01-01")
        start_date = datetime(2020, 1, 1)

    end_date = datetime.today()
    print(f"📅 Fetching data until: {end_date.strftime('%Y-%m-%d')}")

    # Fetch data from yfinance
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,   # Keep Adj Close separate
            group_by=None        # Don't group by ticker for single symbol
        )
    except Exception as e:
        print(f"❌ Failed to fetch data for {symbol}: {e}")
        return

    if df.empty:
        print(f"⚠ No new data available for {symbol}")
        return

    print(f"📊 Downloaded {len(df)} rows of data for {symbol}")
    
    # Insert/update data in database
    inserted_count = upsert_rows(symbol, df)
    if inserted_count > 0:
        print(f"✅ {inserted_count} rows inserted/updated for {symbol}")
    else:
        print(f"❌ Failed to insert data for {symbol}")

# -------------------------
# Main Execution
# -------------------------
def main():
    """
    Main function to process multiple stock symbols.
    Add more symbols to the list as needed.
    """
    symbols = ["AAPL", "GOOGL", "MSFT"]  # Add more symbols here
    
    print("🚀 Starting finance data pipeline...")
    print(f"📈 Processing {len(symbols)} symbols: {', '.join(symbols)}")
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*50}")
        print(f"📊 Processing {symbol} ({i}/{len(symbols)})")
        print(f"{'='*50}")
        
        try:
            fetch_and_store(symbol)
        except Exception as e:
            print(f"❌ Unexpected error processing {symbol}: {e}")
            continue
    
    print(f"\n🎉 Pipeline completed for all symbols!")

if __name__ == "__main__":
    main()