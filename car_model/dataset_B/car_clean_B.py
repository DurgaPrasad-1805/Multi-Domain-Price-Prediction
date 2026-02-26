import pandas as pd
import numpy as np

df = pd.read_csv("germany_used_cars.csv")

# Drop unnecessary columns
df = df.drop([
    "Unnamed: 0",
    "model",
    "registration_date",
    "fuel_consumption_g_km",
    "offer_description"
], axis=1)

# Convert numeric-like columns
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")

# Clean fuel consumption (remove text)
df["fuel_consumption_l_100km"] = (
    df["fuel_consumption_l_100km"]
    .str.replace(" l/100km", "", regex=False)
)

df["fuel_consumption_l_100km"] = pd.to_numeric(
    df["fuel_consumption_l_100km"],
    errors="coerce"
)

# Drop rows with missing target
df = df.dropna(subset=["price_in_euro"])

# Fill numeric missing values
num_cols = df.select_dtypes(include=["float64", "int64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Drop remaining null rows
df = df.dropna()

print("Cleaned Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nFirst 5 rows:")
print(df.head())