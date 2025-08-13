import streamlit as st
import pandas as pd
import mysql.connector
import plotly.graph_objects as go
import plotly.express as px




# ------------------------------
# Database Connection
# ------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="div@MySQL25",  # Change if needed
        database="finance_data"
    )

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data(symbol):
    conn = get_db_connection()
    query = f"SELECT * FROM stocks WHERE symbol = '{symbol}' ORDER BY date"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ------------------------------
# Moving Average Calculation
# ------------------------------
def add_moving_averages(df):
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["MA200"] = df["close"].rolling(window=200).mean()
    return df

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")
st.title("📈 Stock Market Analysis Dashboard")

# Sidebar - stock selection
symbols = ["AAPL", "GOOGL", "MSFT"]
selected_symbols = st.sidebar.multiselect("Select Stocks", symbols, default=["AAPL"])

# Sidebar - date filter
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Main content
for symbol in selected_symbols:
    st.subheader(f"📊 {symbol} Analysis")

    df = load_data(symbol)
    if df.empty:
        st.warning(f"No data found for {symbol}")
        continue

    # Filter by date
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    if df.empty:
        st.warning(f"No data available for {symbol} in selected date range.")
        continue

    df = add_moving_averages(df)

    # Candlestick Chart
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Candlestick"
    )])

    # Add Moving Averages
    fig_candle.add_trace(go.Scatter(x=df["date"], y=df["MA50"], mode='lines', name="MA50"))
    fig_candle.add_trace(go.Scatter(x=df["date"], y=df["MA200"], mode='lines', name="MA200"))

    fig_candle.update_layout(title=f"{symbol} Candlestick Chart with Moving Averages", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_candle, use_container_width=True)

    # Volume Chart
    fig_vol = px.bar(df, x="date", y="volume", title=f"{symbol} Trading Volume")
    st.plotly_chart(fig_vol, use_container_width=True)

    # Data Table
    with st.expander(f"View Raw Data for {symbol}"):
        st.dataframe(df)

st.success("✅ Dashboard loaded successfully!")
