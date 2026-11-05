
import pandas as pd

# Model comparison results
results = {
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree"
    ],
    "Accuracy (%)": [
        86.79,
        88.79,
        84.76
    ],
    "Placed Precision": [
        0.80,
        0.85,
        0.75
    ],
    "Placed Recall": [
        0.76,
        0.76,
        0.74
    ],
    "Placed F1-Score": [
        0.78,
        0.80,
        0.75
    ]
}

df = pd.DataFrame(results)

# Sort models by accuracy
df = df.sort_values(
    by="Accuracy (%)",
    ascending=False
)

# Save comparison report
df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON REPORT")
print(df)

print("\nBest Model:", df.iloc[0]["Model"])
print("Best Accuracy:", df.iloc[0]["Accuracy (%)"], "%")

print("\nReport saved successfully!")
