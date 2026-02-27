import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv(
    "germany_used_cars.csv",
    encoding="latin1",
    engine="python"
)

df = df.drop([
    "Unnamed: 0",
    "model",
    "registration_date",
    "fuel_consumption_g_km",
    "offer_description"
], axis=1)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")

df["fuel_consumption_l_100km"] = (
    df["fuel_consumption_l_100km"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .str.extract(r'(\d+\.?\d*)')[0]
)

df["fuel_consumption_l_100km"] = pd.to_numeric(
    df["fuel_consumption_l_100km"],
    errors="coerce"
)

df["price_in_euro"] = (
    df["price_in_euro"]
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')[0]
)

df["price_in_euro"] = pd.to_numeric(
    df["price_in_euro"],
    errors="coerce"
)

df = df.dropna()

X = df.drop("price_in_euro", axis=1)
y = df["price_in_euro"]

num_cols = X.select_dtypes(include=["float64", "int64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=25,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nCar Dataset B Results")
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

joblib.dump(model, "car_model_B.pkl")
print("Lightweight model saved as car_model_B.pkl")