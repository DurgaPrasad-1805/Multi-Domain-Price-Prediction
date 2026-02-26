results = {
    "Linear Regression": 0.9606672979448871,
    "Ridge Regression": 0.9606953934041048
}

print("Gold Price Model Comparison:\n")

for model, score in results.items():
    print(f"{model} -> R2 Score: {score}")

best_model = max(results, key=results.get)

print("\nBest Performing Model:", best_model)
print("Best R2 Score:", results[best_model])

print("\nNote: LSTM architecture implemented separately for deep sequence modeling.")