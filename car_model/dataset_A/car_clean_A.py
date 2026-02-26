import pandas as pd

df = pd.read_csv("indian_used_cars.csv")

# Drop unnecessary columns
df = df.drop([
    "model",
    "reg_year",
    "overall_cost",
    "has_insurance",
    "spare_key",
    "reg_number",
    "title"
], axis=1)

# Drop rows with missing target
df = df.dropna(subset=["price"])

# Fill missing numerical values
df["engine_capacity(CC)"] = df["engine_capacity(CC)"].fillna(
    df["engine_capacity(CC)"].median()
)

# Drop remaining null rows
df = df.dropna()

print("Cleaned Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nFirst 5 rows:")
print(df.head())