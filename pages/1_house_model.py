import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.title("House Price Prediction")

# --------------------------------------------------
# Dataset Selection
# --------------------------------------------------

dataset_option = st.selectbox(
    "Select Dataset",
    ["California Housing (Dataset A)", "Bengaluru Housing (Dataset B)"]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if dataset_option == "California Housing (Dataset A)":
    file_path = os.path.join(
        BASE_DIR,
        "house_model",
        "dataset_A",
        "california_housing_prices.csv"
    )
    model_path = os.path.join(
        BASE_DIR,
        "house_model",
        "dataset_A",
        "house_model_A.pkl"
    )
    target_column = "median_house_value"

else:
    file_path = os.path.join(
        BASE_DIR,
        "house_model",
        "dataset_B",
        "bengaluru_housing_prices.csv"
    )
    model_path = os.path.join(
        BASE_DIR,
        "house_model",
        "dataset_B",
        "house_model_B.pkl"
    )
    target_column = "price"

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(file_path)

# --------------------------------------------------
# Apply Training Cleaning for Dataset B
# --------------------------------------------------

if dataset_option == "Bengaluru Housing (Dataset B)":

    # Drop society
    if "society" in df.columns:
        df = df.drop("society", axis=1)

    # Extract BHK from size
    if "size" in df.columns:
        df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
        df = df.drop("size", axis=1)

    # Convert total_sqft
    def convert_sqft(x):
        try:
            if "-" in str(x):
                vals = x.split("-")
                return (float(vals[0]) + float(vals[1])) / 2
            return float(x)
        except:
            return None

    if "total_sqft" in df.columns:
        df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

    # Fill missing values
    if "bath" in df.columns:
        df["bath"] = df["bath"].fillna(df["bath"].median())

    if "balcony" in df.columns:
        df["balcony"] = df["balcony"].fillna(df["balcony"].median())

    if "bhk" in df.columns:
        df["bhk"] = df["bhk"].fillna(df["bhk"].median())

    df = df.dropna()

    # Remove outliers
    if "total_sqft" in df.columns and "bhk" in df.columns:
        df = df[df["total_sqft"] / df["bhk"] >= 300]

    if "bath" in df.columns and "bhk" in df.columns:
        df = df[df["bath"] <= df["bhk"] + 2]

    # Reduce rare locations
    if "location" in df.columns:
        location_counts = df["location"].value_counts()
        df["location"] = df["location"].apply(
            lambda x: x if location_counts[x] >= 10 else "other"
        )

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Statistics")

col1, col2 = st.columns(2)

with col1:
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

with col2:
    st.write("Descriptive Statistics:")
    st.write(df.describe())

# --------------------------------------------------
# HISTOGRAMS (Numeric only for visualization)
# --------------------------------------------------

st.subheader("Feature Distributions")

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 10))
numeric_df.hist(bins=30, figsize=(12, 10))
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

st.subheader("Correlation Heatmap")

corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt.gcf())
plt.clf()

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

model = joblib.load(model_path)

# --------------------------------------------------
# Prepare Features (All Features Used)
# --------------------------------------------------

X = df.drop(target_column, axis=1)
y = df[target_column]

st.subheader("Features Used for Prediction")
st.dataframe(pd.DataFrame({"Features": X.columns}))

# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.subheader("Model Performance")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)

st.write(f"R² Score: {round(r2,4)}")

# --------------------------------------------------
# ACTUAL vs PREDICTED
# --------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(predictions[:200], label="Predicted")
plt.legend()
st.pyplot(plt.gcf())
plt.clf()