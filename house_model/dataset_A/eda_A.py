import pandas as pd

# Load dataset
df = pd.read_csv("california_housing_prices.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nColumns:")
print(df.columns)