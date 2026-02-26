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
# Load and Clean Data
# ------------------------

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

# Clean fuel consumption column properly
df["fuel_consumption_l_100km"] = (
    df["fuel_consumption_l_100km"]
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')[0]
)

df["fuel_consumption_l_100km"] = pd.to_numeric(
    df["fuel_consumption_l_100km"],
    errors="coerce"
)

# Clean target column properly
df["price_in_euro"] = (
    df["price_in_euro"]
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')[0]
)

df["price_in_euro"] = pd.to_numeric(
    df["price_in_euro"],
    errors="coerce"
)

# Drop rows where price is missing after cleaning
df = df.dropna(subset=["price_in_euro"])

num_cols_all = df.select_dtypes(include=["float64", "int64"]).columns
for col in num_cols_all:
    df[col] = df[col].fillna(df[col].median())

df = df.dropna()

df = df.sample(n=10000, random_state=42)

# ------------------------
# Split Features & Target
# ------------------------

X = df.drop("price_in_euro", axis=1)
y = df["price_in_euro"]

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

    "Random Forest": RandomForestRegressor(
        n_estimators=50,      # reduce trees
        max_depth=15,         # limit tree depth
        n_jobs=-1,            # use all CPU cores
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=50,      # reduce boosting rounds
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
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

joblib.dump(best_model, "car_model_B.pkl")
print("\nBest Germany car model saved as car_model_B.pkl")