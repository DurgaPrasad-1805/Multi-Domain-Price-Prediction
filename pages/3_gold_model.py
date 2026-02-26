import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Gold Price Prediction")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "gold_model", "gold_price_data.csv")
model_path = os.path.join(BASE_DIR, "gold_model", "gold_model_linear.pkl")

df = pd.read_csv(data_path)

# --------------------------------------------------
# SORT BY DATE (BUT DO NOT USE AS FEATURE)
# --------------------------------------------------

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

# --------------------------------------------------
# FEATURE ENGINEERING (MATCH TRAINING EXACTLY)
# --------------------------------------------------

df['GLD_lag_1'] = df['GLD'].shift(1)
df['GLD_lag_7'] = df['GLD'].shift(7)
df['GLD_lag_30'] = df['GLD'].shift(30)

df['GLD_roll_7'] = df['GLD'].rolling(window=7).mean()
df['GLD_roll_30'] = df['GLD'].rolling(window=30).mean()

df.dropna(inplace=True)

# Remove Date column BEFORE prediction
if "Date" in df.columns:
    df = df.drop("Date", axis=1)

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
# FEATURE DISTRIBUTIONS
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
# PREPARE DATA (MATCH TRAINING)
# --------------------------------------------------

X = df.drop("GLD", axis=1)
y = df["GLD"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")
st.write(f"R² Score: {round(r2,4)}")
st.write("Model Used: Linear Regression with Lag Features")

# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:200], label="Actual GLD")
plt.plot(predictions[:200], label="Predicted GLD")
plt.legend()
st.pyplot(plt.gcf())
plt.clf()