import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# App title
st.title("📈 Stock Price Prediction (Next 30 Days)")
st.markdown("Using Facebook Prophet & Yahoo Finance")

# Dropdown for stock selection
stocks = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA"
}
selected_stock = st.selectbox("Select a stock", list(stocks.keys()))
symbol = stocks[selected_stock]

# Load data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period='5y')
    df.reset_index(inplace=True)
    return df

data = load_data(symbol)

# Show raw data
st.subheader("Raw Data")
st.write(data.tail())

# Plot closing price
def plot_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
    fig.update_layout(title_text="Closing Price Over Time", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_data()

# Prepare data for Prophet
df_train = data[['Date', 'Close']].copy()
df_train.columns = ['ds', 'y']  # Prophet requires these column names

# Ensure types are correct
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train = df_train.dropna()

# Fit Prophet model
model = Prophet(daily_seasonality=True)
model.fit(df_train)

# Forecast for next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Show forecast plot
st.subheader("Forecast Plot")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# Show forecast data
st.subheader("Forecast Data (Next 30 Days)")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
