import pandas as pd
from src.config import DATA_PATH, TARGET, FAILURE_MODES

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

print("\n--- Target ---")
print(df[TARGET].value_counts().to_dict())
print(f"Failure rate: {df[TARGET].mean():.2%}")

print("\n--- Failure modes ---")
print(df[FAILURE_MODES].sum().to_dict())

any_mode = df[FAILURE_MODES].any(axis=1)
print("\n--- Label consistency ---")
print("Failure=1 but no mode flagged:", int(((df[TARGET] == 1) & ~any_mode).sum()))
print("Mode flagged but failure=0:   ", int(((df[TARGET] == 0) & any_mode).sum()))

print("\n--- Machine type ---")
print(df["Type"].value_counts().to_dict())

print("\n--- Statistics ---")
print(df.describe().round(2).to_string())