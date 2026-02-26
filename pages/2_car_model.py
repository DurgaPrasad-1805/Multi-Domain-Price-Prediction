import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Car Price Prediction")

# --------------------------------------------------
# DATASET SELECTION
# --------------------------------------------------

dataset_choice = st.selectbox(
    "Select Dataset",
    ["Indian Used Cars (Dataset A)", "Germany Used Cars (Dataset B)"]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if dataset_choice == "Indian Used Cars (Dataset A)":
    file_path = os.path.join(
        BASE_DIR,
        "car_model",
        "dataset_A",
        "indian_used_cars.csv"
    )
    model_path = os.path.join(
        BASE_DIR,
        "car_model",
        "dataset_A",
        "car_model_A.pkl"
    )
    target_column = "price"

else:
    file_path = os.path.join(
        BASE_DIR,
        "car_model",
        "dataset_B",
        "germany_used_cars.csv"
    )
    model_path = os.path.join(
        BASE_DIR,
        "car_model",
        "dataset_B",
        "car_model_B.pkl"
    )
    target_column = "price_in_euro"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(file_path)

# --------------------------------------------------
# APPLY TRAINING CLEANING
# --------------------------------------------------

if dataset_choice == "Indian Used Cars (Dataset A)":

    df = df.drop([
        "model",
        "reg_year",
        "overall_cost",
        "has_insurance",
        "spare_key",
        "reg_number",
        "title"
    ], axis=1, errors="ignore")

    if "engine_capacity(CC)" in df.columns:
        df["engine_capacity(CC)"] = df["engine_capacity(CC)"].fillna(
            df["engine_capacity(CC)"].median()
        )

    df = df.dropna()

else:

    df = df.drop([
        "Unnamed: 0",
        "model",
        "registration_date",
        "fuel_consumption_g_km",
        "offer_description"
    ], axis=1, errors="ignore")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")

    df["fuel_consumption_l_100km"] = (
        df["fuel_consumption_l_100km"]
        .astype(str)
        .str.extract(r'(\d+\.?\d*)')[0]
    )

    df["fuel_consumption_l_100km"] = pd.to_numeric(
        df["fuel_consumption_l_100km"],
        errors="coerce"
    )

    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.extract(r'(\d+\.?\d*)')[0]
    )

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    df = df.dropna(subset=[target_column])

    num_cols_all = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols_all:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna()

    df = df.sample(n=min(10000, len(df)), random_state=42)

# --------------------------------------------------
# DATA PREVIEW
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
# FEATURE DISTRIBUTIONS (Numeric Only)
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
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load(model_path)

# --------------------------------------------------
# USE ALL FEATURES
# --------------------------------------------------

X = df.drop(target_column, axis=1)
y = df[target_column]

st.subheader("Features Used for Prediction")
st.dataframe(pd.DataFrame({"Features": X.columns}))

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("Model Performance")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)

st.write(f"R² Score: {round(r2,4)}")

# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(predictions[:200], label="Predicted")
plt.legend()
st.pyplot(plt.gcf())
plt.clf()