import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from mysql.connector import Error
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def get_db_connection():
    """
    Establish secure connection to MySQL database with proper error handling.
    """
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="finance_user",
            password="divu@MySQL25",
            database="finance_data",
            autocommit=True
        )
        return connection
    except Error as e:
        print(f"❌ Database connection error: {e}")
        return None

def get_available_symbols():
    """
    Get list of all available stock symbols in the database.
    """
    conn = get_db_connection()
    if conn is None:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM stocks ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return symbols
    except Error as e:
        print(f"❌ Error fetching symbols: {e}")
        return []

def fetch_stock_data(symbol, days_back=None):
    """
    Fetch stock data for a specific symbol with optional date filtering.
    Uses parameterized queries to prevent SQL injection.
    """
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        # Base query with parameterized symbol
        base_query = """
            SELECT date, open, high, low, close, volume
            FROM stocks
            WHERE symbol = %s
        """
        
        params = [symbol]
        
        # Add date filtering if specified
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            base_query += " AND date >= %s"
            params.append(cutoff_date.strftime('%Y-%m-%d'))
        
        base_query += " ORDER BY date"
        
        # Execute query and create DataFrame
        df = pd.read_sql(base_query, conn, params=params)
        conn.close()
        
        if df.empty:
            print(f"⚠️ No data found for symbol: {symbol}")
            return df
        
        # Convert date to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        print(f"📊 Loaded {len(df)} records for {symbol}")
        return df
        
    except Error as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for stock analysis.
    """
    if df.empty:
        return df
    
    # Simple Moving Averages
    df['SMA_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['close'].ewm(span=12).mean()
    df['EMA_26'] = df['close'].ewm(span=26).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Daily Returns
    df['Daily_Return'] = df['close'].pct_change() * 100
    
    # Volatility (30-day rolling standard deviation of returns)
    df['Volatility_30d'] = df['Daily_Return'].rolling(window=30).std()
    
    # Volume Moving Average
    df['Volume_MA_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
    
    return df

def create_comprehensive_analysis_plot(df, symbol):
    """
    Create a comprehensive multi-panel plot showing various aspects of stock analysis.
    """
    if df.empty:
        print("❌ No data to plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.suptitle(f'📈 Comprehensive Analysis: {symbol}', fontsize=16, fontweight='bold')
    
    # Plot 1: Price and Moving Averages with Bollinger Bands
    axes[0].plot(df.index, df['close'], label='Close Price', linewidth=2, color='#2E86AB')
    axes[0].plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.8, color='#A23B72')
    axes[0].plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.8, color='#F18F01')
    
    # Add Bollinger Bands if available
    if 'BB_Upper' in df.columns and df['BB_Upper'].notna().any():
        axes[0].fill_between(df.index, df['BB_Upper'], df['BB_Lower'], 
                           alpha=0.2, color='gray', label='Bollinger Bands')
        axes[0].plot(df.index, df['BB_Upper'], linestyle='--', alpha=0.6, color='gray')
        axes[0].plot(df.index, df['BB_Lower'], linestyle='--', alpha=0.6, color='gray')
    
    axes[0].set_title('Stock Price with Technical Indicators')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Volume Analysis
    axes[1].bar(df.index, df['volume'], alpha=0.6, color='#C73E1D', label='Daily Volume')
    if 'Volume_MA_20' in df.columns:
        axes[1].plot(df.index, df['Volume_MA_20'], color='#2E86AB', 
                    linewidth=2, label='Volume MA 20')
    axes[1].set_title('Trading Volume Analysis')
    axes[1].set_ylabel('Volume')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: MACD Analysis
    if 'MACD' in df.columns and df['MACD'].notna().any():
        axes[2].plot(df.index, df['MACD'], label='MACD', color='#2E86AB')
        axes[2].plot(df.index, df['MACD_Signal'], label='Signal Line', color='#A23B72')
        axes[2].bar(df.index, df['MACD_Histogram'], alpha=0.6, 
                   color='gray', label='MACD Histogram')
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_title('MACD (Moving Average Convergence Divergence)')
        axes[2].set_ylabel('MACD')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: RSI and Daily Returns
    if 'RSI' in df.columns and df['RSI'].notna().any():
        # Create twin axis for RSI
        ax4_twin = axes[3].twinx()
        
        # Daily returns as bars
        colors = ['green' if x > 0 else 'red' for x in df['Daily_Return']]
        axes[3].bar(df.index, df['Daily_Return'], alpha=0.6, color=colors)
        axes[3].set_ylabel('Daily Return (%)', color='black')
        axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # RSI as line
        ax4_twin.plot(df.index, df['RSI'], color='purple', linewidth=2, label='RSI')
        ax4_twin.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax4_twin.axhline(y=30, color='blue', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax4_twin.set_ylabel('RSI', color='purple')
        ax4_twin.set_ylim(0, 100)
        ax4_twin.legend(loc='upper right')
        
        axes[3].set_title('Daily Returns and RSI Analysis')
    else:
        # Just show daily returns if RSI isn't available
        colors = ['green' if x > 0 else 'red' for x in df['Daily_Return']]
        axes[3].bar(df.index, df['Daily_Return'], alpha=0.6, color=colors)
        axes[3].set_title('Daily Returns Analysis')
        axes[3].set_ylabel('Daily Return (%)')
        axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    # Format x-axis dates
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def generate_performance_summary(df, symbol):
    """
    Generate comprehensive performance summary statistics.
    """
    if df.empty:
        return
    
    print(f"\n{'='*60}")
    print(f"📊 PERFORMANCE SUMMARY FOR {symbol}")
    print(f"{'='*60}")
    
    # Basic statistics
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    days_analyzed = len(df)
    
    print(f"📅 Analysis Period: {start_date} to {end_date} ({days_analyzed} trading days)")
    
    # Price statistics
    current_price = df['close'].iloc[-1]
    start_price = df['close'].iloc[0]
    total_return = ((current_price - start_price) / start_price) * 100
    
    print(f"💰 Price Analysis:")
    print(f"   Starting Price: ${start_price:.2f}")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Total Return: {total_return:.2f}%")
    print(f"   Highest Price: ${df['high'].max():.2f}")
    print(f"   Lowest Price: ${df['low'].min():.2f}")
    
    # Return statistics
    if 'Daily_Return' in df.columns:
        avg_daily_return = df['Daily_Return'].mean()
        volatility = df['Daily_Return'].std()
        max_gain = df['Daily_Return'].max()
        max_loss = df['Daily_Return'].min()
        
        print(f"📈 Return Statistics:")
        print(f"   Average Daily Return: {avg_daily_return:.3f}%")
        print(f"   Daily Volatility: {volatility:.3f}%")
        print(f"   Best Single Day: {max_gain:.2f}%")
        print(f"   Worst Single Day: {max_loss:.2f}%")
        
        # Risk-adjusted return (Sharpe ratio approximation)
        if volatility > 0:
            sharpe_approx = (avg_daily_return * np.sqrt(252)) / (volatility * np.sqrt(252))
            print(f"   Annualized Sharpe Ratio: {sharpe_approx:.3f}")
    
    # Volume statistics
    avg_volume = df['volume'].mean()
    max_volume = df['volume'].max()
    print(f"📊 Volume Statistics:")
    print(f"   Average Daily Volume: {avg_volume:,.0f}")
    print(f"   Highest Volume Day: {max_volume:,.0f}")
    
    # Current technical indicators
    if 'RSI' in df.columns and df['RSI'].notna().any():
        current_rsi = df['RSI'].iloc[-1]
        print(f"🔧 Current Technical Indicators:")
        print(f"   RSI: {current_rsi:.1f}")
        
        if current_rsi > 70:
            print(f"   ⚠️  RSI indicates potentially overbought conditions")
        elif current_rsi < 30:
            print(f"   ⚠️  RSI indicates potentially oversold conditions")
        else:
            print(f"   ✅ RSI indicates neutral conditions")

def compare_multiple_stocks(symbols, days_back=365):
    """
    Compare performance of multiple stocks side by side.
    """
    comparison_data = {}
    
    print(f"\n📊 Loading data for comparison analysis...")
    
    for symbol in symbols:
        df = fetch_stock_data(symbol, days_back=days_back)
        if not df.empty:
            # Normalize prices to start at 100 for comparison
            normalized_prices = (df['close'] / df['close'].iloc[0]) * 100
            comparison_data[symbol] = normalized_prices
    
    if not comparison_data:
        print("❌ No data available for comparison")
        return
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    for symbol, prices in comparison_data.items():
        plt.plot(prices.index, prices, label=symbol, linewidth=2)
    
    plt.title(f'📈 Stock Performance Comparison (Last {days_back} Days)\nNormalized to 100 at Start', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Start = 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print(f"📊 COMPARISON SUMMARY (Last {days_back} days)")
    print(f"{'='*60}")
    
    for symbol, prices in comparison_data.items():
        total_return = prices.iloc[-1] - 100
        print(f"{symbol}: {total_return:+.2f}% total return")

def main():
    """
    Main function with interactive menu for stock analysis.
    """
    print("🚀 Welcome to Enhanced Stock Analysis System!")
    print("=" * 50)
    
    # Get available symbols
    available_symbols = get_available_symbols()
    
    if not available_symbols:
        print("❌ No stock symbols found in database. Please run your data pipeline first.")
        return
    
    print(f"📈 Available symbols: {', '.join(available_symbols)}")
    
    while True:
        print(f"\n{'='*50}")
        print("Choose an analysis option:")
        print("1. 📊 Single Stock Comprehensive Analysis")
        print("2. 🔍 Multiple Stock Comparison")
        print("3. 📋 List Available Symbols")
        print("4. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            symbol = input(f"\nEnter stock symbol ({'/'.join(available_symbols)}): ").strip().upper()
            if symbol not in available_symbols:
                print(f"❌ Symbol {symbol} not found in database")
                continue
                
            days_back = input("Enter days back (press Enter for all data): ").strip()
            days_back = int(days_back) if days_back.isdigit() else None
            
            print(f"\n🔄 Analyzing {symbol}...")
            df = fetch_stock_data(symbol, days_back=days_back)
            
            if not df.empty:
                df = calculate_technical_indicators(df)
                create_comprehensive_analysis_plot(df, symbol)
                generate_performance_summary(df, symbol)
            
        elif choice == '2':
            symbols_input = input(f"\nEnter symbols separated by commas ({'/'.join(available_symbols)}): ").strip().upper()
            symbols = [s.strip() for s in symbols_input.split(',')]
            
            # Validate symbols
            valid_symbols = [s for s in symbols if s in available_symbols]
            invalid_symbols = [s for s in symbols if s not in available_symbols]
            
            if invalid_symbols:
                print(f"⚠️  Ignoring invalid symbols: {', '.join(invalid_symbols)}")
            
            if len(valid_symbols) < 2:
                print("❌ Need at least 2 valid symbols for comparison")
                continue
                
            days_back = input("Enter days back for comparison (default 365): ").strip()
            days_back = int(days_back) if days_back.isdigit() else 365
            
            compare_multiple_stocks(valid_symbols, days_back=days_back)
            
        elif choice == '3':
            print(f"\n📋 Available symbols in database:")
            for i, symbol in enumerate(available_symbols, 1):
                print(f"   {i}. {symbol}")
                
        elif choice == '4':
            print("👋 Thank you for using the Stock Analysis System!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()