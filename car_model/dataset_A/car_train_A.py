import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ------------------------
# Load Cleaned Data
# ------------------------

df = pd.read_csv("indian_used_cars.csv")

# Same cleaning steps (repeat for training safety)
df = df.drop([
    "model",
    "reg_year",
    "overall_cost",
    "has_insurance",
    "spare_key",
    "reg_number",
    "title"
], axis=1)

df["engine_capacity(CC)"] = df["engine_capacity(CC)"].fillna(
    df["engine_capacity(CC)"].median()
)

df = df.dropna()

# ------------------------
# Split Features & Target
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

# ------------------------
# Models
# ------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
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
joblib.dump(best_model, "car_model_A.pkl")
print("\nBest car model saved as car_model_A.pkl")