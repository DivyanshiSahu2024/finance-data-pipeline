import os
import mysql.connector
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Load database credentials
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

class DataQualityChecker:
    """
    Data quality checking system for financial data.
    Validates data integrity and logs issues for monitoring.
    """
    
    def __init__(self, log_file_path="logs/data_quality.log"):
        """Initialize the data quality checker with logging configuration."""
        self.log_file_path = log_file_path
        self.setup_logging()
        self.issues_found = []
    
    def setup_logging(self):
        """Configure logging for data quality issues."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure logger
        self.logger = logging.getLogger(f'DataQuality_{datetime.now().strftime("%Y%m%d")}')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for persistent logging
        file_handler = logging.FileHandler(self.log_file_path)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('🔍 %(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def validate_row(self, row: pd.Series, symbol: str, date: str) -> Tuple[bool, List[str]]:
        """
        Validate individual row of stock price data.
        
        Returns:
            Tuple of (is_valid: bool, error_messages: List[str])
        """
        errors = []
        
        # Check for missing or null values
        required_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        for field in required_fields:
            if field not in row or pd.isna(row[field]) or row[field] is None:
                errors.append(f"Missing {field}")
        
        if errors:  # If we have missing data, return early
            return False, errors
        
        # Extract values for validation
        open_val = float(row['Open'])
        high_val = float(row['High'])
        low_val = float(row['Low'])
        close_val = float(row['Close'])
        volume_val = int(row['Volume']) if not pd.isna(row['Volume']) else 0
        
        # Basic range validations
        if open_val <= 0:
            errors.append(f"Invalid open price: ${open_val}")
        
        if high_val <= 0:
            errors.append(f"Invalid high price: ${high_val}")
        
        if low_val <= 0:
            errors.append(f"Invalid low price: ${low_val}")
        
        if close_val <= 0:
            errors.append(f"Invalid close price: ${close_val}")
        
        if volume_val < 0:  # Volume can be 0 for some stocks/days
            errors.append(f"Negative volume: {volume_val:,}")
        
        # Logical price relationships
        if high_val < low_val:
            errors.append(f"High (${high_val}) < Low (${low_val})")
        
        if high_val < max(open_val, close_val):
            errors.append(f"High (${high_val}) < Open/Close max")
        
        if low_val > min(open_val, close_val):
            errors.append(f"Low (${low_val}) > Open/Close min")
        
        # Extreme price movement validation (>50% change in one day is suspicious)
        if abs(close_val - open_val) / open_val > 0.50:
            change_pct = ((close_val - open_val) / open_val) * 100
            errors.append(f"Extreme price movement: {change_pct:+.1f}% in one day")
        
        # Volume validation for major stocks
        if volume_val == 0 and symbol in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']:
            errors.append(f"Zero volume for major stock on trading day")
        
        return len(errors) == 0, errors
    
    def check_dataframe_quality(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Check data quality for entire DataFrame and return cleaned version.
        
        Args:
            df: DataFrame containing stock data
            symbol: Stock symbol being processed
        
        Returns:
            Cleaned DataFrame with invalid rows removed
        """
        if df.empty:
            self.logger.warning(f"Empty dataset provided for {symbol}")
            return df
        
        original_count = len(df)
        self.logger.info(f"Starting quality check for {symbol} ({original_count} rows)")
        
        valid_indices = []
        invalid_count = 0
        
        for index, row in df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d') if 'Date' in row else str(index)
            
            is_valid, error_messages = self.validate_row(row, symbol, date_str)
            
            if is_valid:
                valid_indices.append(index)
            else:
                invalid_count += 1
                error_summary = " | ".join(error_messages)
                self.logger.error(f"{symbol} | {date_str} | {error_summary}")
                self.issues_found.extend(error_messages)
        
        # Create cleaned DataFrame
        cleaned_df = df.loc[valid_indices].copy()
        
        # Log summary
        if invalid_count > 0:
            success_rate = (len(cleaned_df) / original_count) * 100
            self.logger.warning(f"Quality Summary for {symbol}: "
                              f"{len(cleaned_df)}/{original_count} rows passed "
                              f"({success_rate:.1f}% success rate)")
        else:
            self.logger.info(f"✅ All {original_count} rows passed quality checks for {symbol}")
        
        return cleaned_df
    
    def get_quality_summary(self) -> str:
        """Generate a summary of quality issues found."""
        if not self.issues_found:
            return "✅ No data quality issues found!"
        
        # Count issue types
        issue_counts = {}
        for issue in self.issues_found:
            issue_type = issue.split(':')[0].split('(')[0].strip()
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        summary = f"⚠️ Found {len(self.issues_found)} data quality issues:\n"
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            summary += f"   • {issue_type}: {count} occurrences\n"
        
        return summary.strip()

def get_db_connection():
    """Establish connection to MySQL database."""
    try:
        return mysql.connector.connect(
            host="127.0.0.1",   # Force TCP instead of named pipe
            port=3306,          # Default MySQL port
            user="root",
            password="div@MySQL25",
            database="finance_data"
        )
    except mysql.connector.Error as e:
        print(f"❌ Database connection error: {e}")
        return None

def get_last_close_price(symbol):
    """Get the last closing price for a symbol to check data continuity."""
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT close FROM stocks 
            WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
        """, (symbol,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return float(result[0]) if result else None
    except Exception as e:
        print(f"⚠️ Error getting last price for {symbol}: {e}")
        return None

def fetch_stock_data(symbol, period="5d"):
    """
    Fetch stock data from Yahoo Finance with error handling.
    
    Args:
        symbol: Stock symbol to fetch
        period: Time period (5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        DataFrame with stock data or empty DataFrame if error
    """
    try:
        print(f"📥 Fetching {period} data for {symbol} from Yahoo Finance...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            print(f"⚠️ No data returned for {symbol}")
            return df
        
        # Reset index to make Date a regular column
        df.reset_index(inplace=True)
        print(f"✅ Successfully fetched {len(df)} rows for {symbol}")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def store_stock_data_with_quality_check(symbol, df):
    """
    Store stock data with comprehensive quality checking.
    
    Args:
        symbol: Stock symbol
        df: DataFrame containing stock data
    
    Returns:
        Number of rows successfully stored
    """
    if df.empty:
        print(f"⚠️ No data to store for {symbol}")
        return 0
    
    # Initialize quality checker
    quality_checker = DataQualityChecker()
    
    # Get last known price for continuity checking
    last_price = get_last_close_price(symbol)
    if last_price:
        print(f"📊 Last known price for {symbol}: ${last_price:.2f}")
    
    # Run quality checks
    print(f"🔍 Running data quality checks for {symbol}...")
    cleaned_df = quality_checker.check_dataframe_quality(df, symbol)
    
    if cleaned_df.empty:
        print(f"❌ All data for {symbol} failed quality checks")
        print(quality_checker.get_quality_summary())
        return 0
    
    # Check for data continuity if we have historical data
    if last_price and not cleaned_df.empty:
        first_new_price = cleaned_df.iloc[0]['Open']
        price_ratio = first_new_price / last_price
        
        if price_ratio < 0.5 or price_ratio > 2.0:
            print(f"⚠️ Potential price discontinuity detected:")
            print(f"   Last price: ${last_price:.2f}")
            print(f"   New price: ${first_new_price:.2f}")
            print(f"   Ratio: {price_ratio:.2f}x")
            print("   This could indicate a stock split or data issue")
    
    # Store the cleaned data
    conn = get_db_connection()
    if conn is None:
        return 0
    
    try:
        cursor = conn.cursor()
        stored_count = 0
        
        for _, row in cleaned_df.iterrows():
            cursor.execute("""
                INSERT INTO stocks (symbol, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    open=VALUES(open),
                    high=VALUES(high),
                    low=VALUES(low),
                    close=VALUES(close),
                    volume=VALUES(volume)
            """, (
                symbol,
                row["Date"].date(),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"])
            ))
            stored_count += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Report results
        rows_removed = len(df) - len(cleaned_df)
        if rows_removed > 0:
            print(f"⚠️ Removed {rows_removed} invalid rows during quality check")
            print(quality_checker.get_quality_summary())
        
        print(f"✅ Successfully stored {stored_count} quality-checked rows for {symbol}")
        return stored_count
        
    except mysql.connector.Error as e:
        print(f"❌ Database error while storing {symbol}: {e}")
        conn.rollback()
        return 0
    finally:
        if conn.is_connected():
            conn.close()

def main():
    """
    Main function with enhanced error handling and user interaction.
    """
    print("🚀 Enhanced Stock Data Pipeline with Quality Checking")
    print("=" * 55)
    
    # Get stock symbol from user or use default
    default_symbol = "AAPL"
    symbol_input = input(f"Enter stock symbol (default: {default_symbol}): ").strip().upper()
    stock_symbol = symbol_input if symbol_input else default_symbol
    
    # Get time period from user or use default
    periods = {
        '1': '5d',
        '2': '1mo', 
        '3': '3mo',
        '4': '6mo',
        '5': '1y',
        '6': '2y',
        '7': 'max'
    }
    
    print("\nAvailable time periods:")
    for key, period in periods.items():
        print(f"  {key}. {period}")
    
    period_choice = input("Select period (1-7, default: 1 for 5d): ").strip()
    selected_period = periods.get(period_choice, '5d')
    
    print(f"\n📈 Processing {stock_symbol} for period: {selected_period}")
    print("-" * 50)
    
    # Fetch and store data with quality checking
    data = fetch_stock_data(stock_symbol, period=selected_period)
    
    if not data.empty:
        rows_stored = store_stock_data_with_quality_check(stock_symbol, data)
        
        if rows_stored > 0:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"   Symbol: {stock_symbol}")
            print(f"   Period: {selected_period}")
            print(f"   Rows stored: {rows_stored}")
        else:
            print(f"\n❌ Pipeline failed - no data was stored")
    else:
        print(f"\n❌ Pipeline failed - no data could be fetched")

if __name__ == "__main__":
    main()