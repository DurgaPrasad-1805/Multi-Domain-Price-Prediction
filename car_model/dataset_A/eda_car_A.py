import pandas as pd

df = pd.read_csv("indian_used_cars.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nColumns:")
print(df.columns)

print("\nUnique Brands:")
if "brand" in df.columns:
    print(df["brand"].nunique())