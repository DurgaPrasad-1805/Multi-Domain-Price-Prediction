import streamlit as st
import pandas as pd
import os
import joblib

st.title("House Price Prediction")

dataset_option = st.selectbox(
    "Select Dataset",
    ["California Housing (Dataset A)", "Bengaluru Housing (Dataset B)"]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if dataset_option == "California Housing (Dataset A)":
    model_path = os.path.join(BASE_DIR, "house_model", "dataset_A", "house_model_A.pkl")
else:
    model_path = os.path.join(BASE_DIR, "house_model", "dataset_B", "house_model_B.pkl")

model = joblib.load(model_path)

st.markdown("---")

if dataset_option == "California Housing (Dataset A)":

    st.subheader("Enter Property Details")

    median_income = st.number_input("Median Income", min_value=0.0, value=5.0)
    housing_median_age = st.number_input("Housing Median Age", min_value=1.0, value=20.0)
    total_rooms = st.number_input("Total Rooms", min_value=1.0, value=1000.0)
    population = st.number_input("Population", min_value=1.0, value=1000.0)
    ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

    if st.button("Predict Price"):
        input_data = pd.DataFrame({
            "longitude": [-122.0],
            "latitude": [37.0],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [500],
            "population": [population],
            "households": [300],
            "median_income": [median_income],
            "ocean_proximity": [ocean_proximity]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: ${round(prediction,2)}")

else:

    st.subheader("Enter Property Details")

    total_sqft = st.number_input("Total Square Feet", min_value=300.0, value=1200.0)
    bhk = st.number_input("BHK", min_value=1.0, value=2.0)
    bath = st.number_input("Bathrooms", min_value=1.0, value=2.0)
    balcony = st.number_input("Balcony", min_value=0.0, value=1.0)
    location = st.text_input("Location", value="Whitefield")

    if st.button("Predict Price"):
        input_data = pd.DataFrame({
            "area_type": ["Super built-up  Area"],
            "availability": ["Ready To Move"],
            "location": [location],
            "total_sqft": [total_sqft],
            "bath": [bath],
            "balcony": [balcony],
            "bhk": [bhk]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: ₹ {round(prediction,2)} Lakhs")