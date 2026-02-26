import streamlit as st
import pandas as pd
import os
import joblib
import re

st.title("Car Price Prediction")

dataset_choice = st.selectbox(
    "Select Dataset",
    ["Indian Used Cars (Dataset A)", "Germany Used Cars (Dataset B)"]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# LOAD MODEL + DATA
# =====================================================

if dataset_choice == "Indian Used Cars (Dataset A)":
    model_path = os.path.join(BASE_DIR, "car_model", "dataset_A", "car_model_A.pkl")
    data_path = os.path.join(BASE_DIR, "car_model", "dataset_A", "indian_used_cars.csv")
    target_column = "price"
    currency_symbol = "₹"
else:
    model_path = os.path.join(BASE_DIR, "car_model", "dataset_B", "car_model_B.pkl")
    data_path = os.path.join(BASE_DIR, "car_model", "dataset_B", "germany_used_cars.csv")
    target_column = "price_in_euro"
    currency_symbol = "€"

model = joblib.load(model_path)
df_original = pd.read_csv(data_path)

# =====================================================
# APPLY SAME CLEANING AS TRAINING (GERMANY ONLY)
# =====================================================

if dataset_choice == "Germany Used Cars (Dataset B)":

    df_original = df_original.drop([
        "Unnamed: 0",
        "model",
        "registration_date",
        "fuel_consumption_g_km",
        "offer_description"
    ], axis=1, errors="ignore")

    df_original["year"] = pd.to_numeric(df_original["year"], errors="coerce")
    df_original["power_kw"] = pd.to_numeric(df_original["power_kw"], errors="coerce")

    # Clean fuel consumption
    df_original["fuel_consumption_l_100km"] = (
        df_original["fuel_consumption_l_100km"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r'(\d+\.?\d*)')[0]
    )
    df_original["fuel_consumption_l_100km"] = pd.to_numeric(
        df_original["fuel_consumption_l_100km"], errors="coerce"
    )

    # Clean price column
    df_original["price_in_euro"] = (
        df_original["price_in_euro"]
        .astype(str)
        .str.extract(r'(\d+\.?\d*)')[0]
    )
    df_original["price_in_euro"] = pd.to_numeric(
        df_original["price_in_euro"], errors="coerce"
    )

    df_original = df_original.dropna(subset=["price_in_euro"])
    df_original = df_original.fillna(df_original.median(numeric_only=True))
    df_original = df_original.dropna()

# Remove target column
df_original = df_original.drop(columns=[target_column], errors="ignore")

# Take clean template row
template_row = df_original.iloc[[0]].copy()

st.markdown("---")
st.subheader("Enter Car Details")

# =====================================================
# DATASET A
# =====================================================

if dataset_choice == "Indian Used Cars (Dataset A)":

    brand = st.selectbox("Brand", df_original["brand"].dropna().unique())
    transmission = st.selectbox("Transmission", df_original["transmission"].dropna().unique())
    fuel_type = st.selectbox("Fuel Type", df_original["fuel_type"].dropna().unique())
    make_year = st.number_input("Make Year", 1995, 2025, int(template_row["make_year"].values[0]))
    engine_capacity = st.number_input("Engine Capacity (CC)", 600.0, 5000.0, float(template_row["engine_capacity(CC)"].values[0]))
    km_driven = st.number_input("Kilometers Driven", 0.0, 500000.0, float(template_row["km_driven"].values[0]))
    ownership = st.selectbox("Ownership", df_original["ownership"].dropna().unique())

    if st.button("Predict Car Price"):

        input_df = template_row.copy()
        input_df["brand"] = brand
        input_df["transmission"] = transmission
        input_df["fuel_type"] = fuel_type
        input_df["make_year"] = make_year
        input_df["engine_capacity(CC)"] = engine_capacity
        input_df["km_driven"] = km_driven
        input_df["ownership"] = ownership

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Car Price: {currency_symbol} {round(prediction,2)}")

# =====================================================
# DATASET B
# =====================================================

else:

    brand = st.selectbox("Brand", df_original["brand"].dropna().unique())
    color = st.selectbox("Color", df_original["color"].dropna().unique())
    year = st.number_input("Manufacturing Year", 1995, 2025, int(template_row["year"].values[0]))
    power_kw = st.number_input("Power (kW)", 20.0, 500.0, float(template_row["power_kw"].values[0]))
    power_ps = st.number_input("Power (PS)", 20.0, 800.0, float(template_row["power_ps"].values[0]))
    transmission_type = st.selectbox("Transmission Type", df_original["transmission_type"].dropna().unique())
    fuel_type = st.selectbox("Fuel Type", df_original["fuel_type"].dropna().unique())
    mileage = st.number_input("Mileage (km)", 0.0, 500000.0, float(template_row["mileage_in_km"].values[0]))
    fuel_consumption = st.number_input("Fuel Consumption (L/100km)", 2.0, 30.0, float(template_row["fuel_consumption_l_100km"].values[0]))

    if st.button("Predict Car Price"):

        input_df = template_row.copy()
        input_df["brand"] = brand
        input_df["color"] = color
        input_df["year"] = year
        input_df["power_kw"] = power_kw
        input_df["power_ps"] = power_ps
        input_df["transmission_type"] = transmission_type
        input_df["fuel_type"] = fuel_type
        input_df["mileage_in_km"] = mileage
        input_df["fuel_consumption_l_100km"] = fuel_consumption

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Car Price: {currency_symbol} {round(prediction,2)}")