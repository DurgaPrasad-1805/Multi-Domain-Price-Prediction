print("\n==============================")
print(" Multi-Domain Price Prediction")
print("==============================\n")

results = {
    "House Model (Best Dataset)": 0.8169,     # replace if needed
    "Car Model (Best Dataset)": 0.7423,       # replace with your best R2
    "Gold Model (Ridge Regression)": 0.9607
}

for domain, score in results.items():
    print(f"{domain} -> R2 Score: {score}")

best_domain = max(results, key=results.get)

print("\n------------------------------")
print("Best Performing Domain:", best_domain)
print("Highest R2 Score:", results[best_domain])
print("------------------------------\n")