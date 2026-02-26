import pandas as pd
import numpy as np

df = pd.read_csv("bengaluru_housing_prices.csv")

# Drop unnecessary column
df = df.drop("society", axis=1)

# Drop rows with missing target
df = df.dropna(subset=["price"])

# Clean size column → extract number of bedrooms
df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)

# Function to convert total_sqft ranges into single value
def convert_sqft(x):
    try:
        if "-" in str(x):
            vals = x.split("-")
            return (float(vals[0]) + float(vals[1])) / 2
        return float(x)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# Drop original size column
df = df.drop("size", axis=1)

# Handle missing values
df["bath"] = df["bath"].fillna(df["bath"].median())
df["balcony"] = df["balcony"].fillna(df["balcony"].median())
df["bhk"] = df["bhk"].fillna(df["bhk"].median())

# Drop remaining null rows
df = df.dropna()

print("Cleaned Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# Remove extreme sqft per bhk (less than 300 sqft per bhk)
df = df[df["total_sqft"] / df["bhk"] >= 300]

# Remove extreme bath count (bath > bhk + 2)
df = df[df["bath"] <= df["bhk"] + 2]

print("\nAfter Outlier Removal Shape:", df.shape)

# Reduce rare locations
location_counts = df["location"].value_counts()

df["location"] = df["location"].apply(
    lambda x: x if location_counts[x] >= 10 else "other"
)

print("\nUnique locations after grouping:", df["location"].nunique())