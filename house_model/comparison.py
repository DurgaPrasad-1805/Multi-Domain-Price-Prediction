results = {
    "California Dataset (A)": 0.8169,
    "Bengaluru Dataset (B)": 0.6664
}

print("House Price Model Comparison:")
for dataset, score in results.items():
    print(f"{dataset} -> R2 Score: {score}")