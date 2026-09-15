import pandas as pd
from src.config import DATA_PATH, TARGET
from src.features import get_X_y

df = pd.read_csv(DATA_PATH)
X, y = get_X_y(df)

print("Feature matrix shape:", X.shape)
print("Features:", list(X.columns))
print("\nMissing values:", int(X.isna().sum().sum()))
print("Infinite values:", int((X.abs() == float("inf")).sum().sum()))

print("\n--- Feature statistics ---")
print(X.describe().round(2).to_string())

# Check engineered features have clear separation
print("\n--- Mean values: Healthy vs Failure ---")
combined = X.copy()
combined["target"] = y.values
grouped = combined.groupby("target").mean().T
grouped.columns = ["Healthy", "Failure"]
grouped["Diff %"] = ((grouped["Failure"] - grouped["Healthy"]) / (grouped["Healthy"].abs() + 1e-9) * 100).round(1)
print(grouped.round(2).to_string())