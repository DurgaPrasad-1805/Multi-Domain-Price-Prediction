import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ------------------------
# Load and Clean Data
# ------------------------

df = pd.read_csv("bengaluru_housing_prices.csv")

# Drop society
df = df.drop("society", axis=1)

# Extract BHK
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

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# Fill missing values
df["bath"] = df["bath"].fillna(df["bath"].median())
df["balcony"] = df["balcony"].fillna(df["balcony"].median())
df["bhk"] = df["bhk"].fillna(df["bhk"].median())

df = df.dropna()

# Remove outliers
df = df[df["total_sqft"] / df["bhk"] >= 300]
df = df[df["bath"] <= df["bhk"] + 2]

# Reduce rare locations
location_counts = df["location"].value_counts()
df["location"] = df["location"].apply(
    lambda x: x if location_counts[x] >= 10 else "other"
)

# ------------------------
# Train Models
# ------------------------

X = df.drop("price", axis=1)
y = df["price"]

num_cols = X.select_dtypes(include=["float64", "int64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

best_model = None
best_r2 = 0

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("MSE:", mean_squared_error(y_test, predictions))
    print("R2 Score:", r2)

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline

# Save best model
joblib.dump(best_model, "house_model_B.pkl")

print("\nBest Dataset B model saved as house_model_B.pkl")