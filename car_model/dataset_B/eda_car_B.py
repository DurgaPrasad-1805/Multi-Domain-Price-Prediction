import pandas as pd

df = pd.read_csv("germany_used_cars.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nColumns:")
print(df.columns)

if "brand" in df.columns:
    print("\nUnique Brands:", df["brand"].nunique())