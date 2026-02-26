results = {
    "India Dataset (A)": 0.7423,
    "Germany Dataset (B)": 0.6895
}

print("Car Price Model Comparison:\n")

for dataset, score in results.items():
    print(f"{dataset} -> R2 Score: {score}")

best_dataset = max(results, key=results.get)

print("\nBest Performing Dataset:", best_dataset)
print("Best R2 Score:", results[best_dataset])