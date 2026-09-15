import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import DATA_PATH, SEED, TARGET
from src.features import get_X_y

# ── Load and split ──
df = pd.read_csv(DATA_PATH)
X, y = get_X_y(df)

# class weight ratio for models that need it
spw = int((y == 0).sum() / (y == 1).sum())
print(f"Class ratio (healthy:failure) = {spw}:1")
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}\n")

# ── Define models ──
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=400, class_weight="balanced",
        random_state=SEED, n_jobs=-1),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        random_state=SEED),

    "XGBoost": XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=SEED, n_jobs=-1, verbosity=0),

    "LightGBM": LGBMClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        scale_pos_weight=spw, random_state=SEED,
        n_jobs=-1, verbose=-1),
}

# ── Cross-validate each model ──
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ["f1", "recall", "precision", "roc_auc", "average_precision"]

results = []
for name, model in models.items():
    print(f"Training {name}...")
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    row = {"Model": name}
    for metric in scoring:
        key = f"test_{metric}"
        mean = scores[key].mean()
        std = scores[key].std()
        row[metric] = f"{mean:.3f} ± {std:.3f}"
        row[f"{metric}_mean"] = mean
    results.append(row)
    print(f"  F1={row['f1']}  Recall={row['recall']}  "
          f"Precision={row['precision']}  ROC-AUC={row['roc_auc']}")

# ── Summary table ──
results_df = pd.DataFrame(results)
display_cols = ["Model", "f1", "recall", "precision", "roc_auc", "average_precision"]
print("\n" + "=" * 90)
print("MODEL COMPARISON (5-Fold Stratified CV)")
print("=" * 90)
print(results_df[display_cols].to_string(index=False))

# ── Identify best model ──
best_idx = max(range(len(results)), key=lambda i: results[i]["f1_mean"])
best_name = results[best_idx]["Model"]
print(f"\n🏆 Best model by F1: {best_name}")
print(f"   F1={results[best_idx]['f1']}  Recall={results[best_idx]['recall']}")

# ── Save comparison ──
from src.config import REPORT_DIR
REPORT_DIR.mkdir(exist_ok=True)
results_df[display_cols].to_csv(REPORT_DIR / "model_comparison.csv", index=False)
print(f"\nResults saved to {REPORT_DIR / 'model_comparison.csv'}")