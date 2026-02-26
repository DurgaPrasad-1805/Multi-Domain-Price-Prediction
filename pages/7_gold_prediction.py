import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.title("Gold Price Prediction")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "gold_model", "gold_price_data.csv")
model_path = os.path.join(BASE_DIR, "gold_model", "gold_model_linear.pkl")

# Load model
model = joblib.load(model_path)

# Load data
df = pd.read_csv(data_path)

# Sort by Date
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Create lag & rolling features (same as training)
df['GLD_lag_1'] = df['GLD'].shift(1)
df['GLD_lag_7'] = df['GLD'].shift(7)
df['GLD_lag_30'] = df['GLD'].shift(30)

df['GLD_roll_7'] = df['GLD'].rolling(window=7).mean()
df['GLD_roll_30'] = df['GLD'].rolling(window=30).mean()

df.dropna(inplace=True)

st.markdown("---")
st.subheader("Enter Market Indicators")

# User Inputs
spx = st.number_input("S&P 500 Index (SPX)", value=float(df["SPX"].iloc[-1]))
uso = st.number_input("US Oil Fund (USO)", value=float(df["USO"].iloc[-1]))
slv = st.number_input("Silver ETF (SLV)", value=float(df["SLV"].iloc[-1]))
eur_usd = st.number_input("EUR/USD Exchange Rate", value=float(df["EUR/USD"].iloc[-1]))

if st.button("Predict Next Gold Price"):

    # Take last row as base
    last_row = df.iloc[-1:].copy()

    # Update with user inputs
    last_row["SPX"] = spx
    last_row["USO"] = uso
    last_row["SLV"] = slv
    last_row["EUR/USD"] = eur_usd

    # Prepare features
    X = last_row.drop(["Date", "GLD"], axis=1)

    prediction = model.predict(X)[0]

    st.success(f"Predicted Gold Price (Next Day): ${round(prediction,2)}")