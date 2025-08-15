import yfinance as yf
import mysql.connector
import pandas as pd

def get_db_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="finance_user",
        password="divu@MySQL25",
        database="finance_data"
    )

def fetch_yfinance_data(symbol, start_date="2024-01-01", end_date="2024-12-31"):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)  # Date becomes a column
    return data

def insert_into_db(symbol, df):
    conn = get_db_connection()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT IGNORE INTO stocks (symbol, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                row['Date'].date(),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                int(row['Volume'])
            ))
        except Exception as e:
            print(f"Error inserting row: {e}")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    ticker = "AAPL"
    df = fetch_yfinance_data(ticker, "2024-01-01", "2024-06-30")
    insert_into_db(ticker, df)
    print(f"Data for {ticker} inserted from Yahoo Finance.")
