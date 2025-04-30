#python minor project 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime
import time
from sklearn.preprocessing import MinMaxScaler


st.markdown("""
    <style>
        .header {
            text-align: center;
            color: #1E90FF;
            font-size: 40px;
        }
        .subheader {
            color: #333333;
        }
        .warning {
            color: #ff5733;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# Title 
st.markdown("<h1 class='header'>📈 Stock Market Trend Prediction</h1>", unsafe_allow_html=True)

# Sidebar 
st.sidebar.header('Stock Market Prediction Settings')

# User input
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))

# stock info
def display_stock_info(ticker):
    stock_info = yf.Ticker(ticker).info
    st.subheader(f"Information for {ticker}")
    st.write(f"**Company Name**: {stock_info.get('longName', 'N/A')}")
    st.write(f"**Sector**: {stock_info.get('sector', 'N/A')}")
    st.write(f"**Market Cap**: {stock_info.get('marketCap', 'N/A')}")
    st.write(f"**PE Ratio**: {stock_info.get('trailingPE', 'N/A')}")
    st.write(f"**Dividend Yield**: {stock_info.get('dividendYield', 'N/A') * 100 if stock_info.get('dividendYield') else 'N/A'}%")

# Display stock info
display_stock_info(user_input)

# Loading Spinner
with st.spinner('Fetching Data and Model...'):
    try:
        # Download stock data from Yahoo Finance
        df = yf.download(user_input, start=start_date, end=end_date)

        # empty data and missing values
        if df.empty:
            st.warning(f"Could not retrieve data for {user_input}. Please check the stock ticker symbol or date range.")
        else:
            df = df.fillna(method='ffill').fillna(method='bfill')  # Forward fill and backfill for missing data

            # correct data types 
            df = df.apply(pd.to_numeric, errors='ignore')

            # Data Overviewe
            st.subheader(f"Data Overview for {user_input}")
            st.dataframe(df.head(), use_container_width=True)

            # Visualizing 
            st.subheader(f'{user_input} Closing Price vs Time Chart')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], label='Closing Price')
            plt.title(f'{user_input} Closing Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            st.pyplot(fig)

            # 100-Day Moving Average 
            st.subheader(f'{user_input} Closing Price with 100-Day Moving Average')
            ma100 = df['Close'].rolling(100).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], label='Closing Price')
            plt.plot(ma100, label='100-Day Moving Average', color='orange')
            plt.title(f'{user_input} Closing Price with 100MA')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig)

            # 100 and 200-Day Moving Averages 
            st.subheader(f'{user_input} Closing Price with 100MA & 200MA')
            ma200 = df['Close'].rolling(200).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], label='Closing Price')
            plt.plot(ma100, label='100-Day Moving Average', color='orange')
            plt.plot(ma200, label='200-Day Moving Average', color='green')
            plt.title(f'{user_input} Closing Price with 100MA & 200MA')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig)

            # Split into training and testing datasets
            data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Load pre-trained model
            model = load_model('stock_keras.h5')

            # Prepare data for predictions
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)

            # Rescale predictions
            scaler = scaler.scale_
            scale_factor = 1 / scaler[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # predictions vs original 
            st.subheader(f'{user_input} Predictions vs Original')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.title(f'{user_input} Predictions vs Original')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig)

            # Progress Bar 
            st.subheader("Loading Prediction Model...")
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.02)

    # errors 
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.warning("Please check the stock ticker symbol or try a different date range.")
