import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv("bengaluru_housing_prices.csv")

df = df.drop("society", axis=1)

df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
df = df.drop("size", axis=1)

def convert_sqft(x):
    try:
        if "-" in str(x):
            vals = x.split("-")
            return (float(vals[0]) + float(vals[1])) / 2
        return float(x)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

df["bath"] = df["bath"].fillna(df["bath"].median())
df["balcony"] = df["balcony"].fillna(df["balcony"].median())
df["bhk"] = df["bhk"].fillna(df["bhk"].median())

df = df.dropna()

df = df[df["total_sqft"] / df["bhk"] >= 300]
df = df[df["bath"] <= df["bhk"] + 2]

location_counts = df["location"].value_counts()
df["location"] = df["location"].apply(
    lambda x: x if location_counts[x] >= 10 else "other"
)

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

model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=25,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nHouse Dataset B Results")
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

joblib.dump(model, "house_model_B.pkl")
print("Lightweight model saved as house_model_B.pkl")