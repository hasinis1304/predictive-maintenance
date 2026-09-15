import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict, train_test_split)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, f1_score)
import matplotlib.pyplot as plt

from src.config import DATA_PATH, SEED, TEST_SIZE, TARGET, MODEL_DIR, REPORT_DIR
from src.features import get_X_y

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ── Load and hold out test set ──
df = pd.read_csv(DATA_PATH)
X, y = get_X_y(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

print(f"Train: {X_train.shape[0]} samples ({y_train.sum()} failures)")
print(f"Test:  {X_test.shape[0]} samples ({y_test.sum()} failures)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ══════════════════════════════════════════
# PART 1: Hyperparameter tuning with Optuna
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("PART 1: Hyperparameter Tuning (this takes 5-10 min)")
print("=" * 50)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": SEED,
    }
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring="f1", n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=80, show_progress_bar=True)

print(f"\nBest CV F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# ══════════════════════════════════════════
# PART 2: Train best model on full training set
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("PART 2: Training Best Model")
print("=" * 50)

best_params = study.best_params
best_params["random_state"] = SEED
best_model = GradientBoostingClassifier(**best_params)
best_model.fit(X_train, y_train)
print("Model trained on full training set.")

# ══════════════════════════════════════════
# PART 3: Threshold optimization
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("PART 3: Threshold Optimization")
print("=" * 50)

# Use out-of-fold predictions on training data (not test set)
oof_probs = cross_val_predict(
    GradientBoostingClassifier(**best_params),
    X_train, y_train, cv=cv, method="predict_proba")[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_threshold = float(thresholds[f1_scores[:-1].argmax()])
best_f1_at_threshold = f1_scores[:-1].max()

print(f"Default threshold (0.5):  F1 = {f1_score(y_train, (oof_probs >= 0.5).astype(int)):.4f}")
print(f"Optimal threshold ({best_threshold:.3f}): F1 = {best_f1_at_threshold:.4f}")

# Plot precision-recall tradeoff
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(thresholds, precisions[:-1], "b-", label="Precision", linewidth=2)
ax1.plot(thresholds, recalls[:-1], "r-", label="Recall", linewidth=2)
ax1.plot(thresholds, f1_scores[:-1], "g--", label="F1", linewidth=2)
ax1.axvline(x=best_threshold, color="gray", linestyle=":", linewidth=1.5,
            label=f"Optimal threshold = {best_threshold:.3f}")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Score")
ax1.set_title("Precision-Recall-F1 vs Decision Threshold")
ax1.legend(loc="best")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR / "07_threshold_optimization.png", dpi=150)
plt.close()
print("Threshold plot saved.")

# ══════════════════════════════════════════
# PART 4: Final evaluation on TEST set
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("PART 4: Final Test Set Evaluation")
print("=" * 50)

test_probs = best_model.predict_proba(X_test)[:, 1]

# With default threshold
y_pred_default = (test_probs >= 0.5).astype(int)
print("\n--- Default threshold (0.50) ---")
print(confusion_matrix(y_test, y_pred_default))
print(classification_report(y_test, y_pred_default, digits=3,
                            target_names=["Healthy", "Failure"]))

# With optimized threshold
y_pred_opt = (test_probs >= best_threshold).astype(int)
print(f"\n--- Optimized threshold ({best_threshold:.3f}) ---")
print(confusion_matrix(y_test, y_pred_opt))
print(classification_report(y_test, y_pred_opt, digits=3,
                            target_names=["Healthy", "Failure"]))

# ══════════════════════════════════════════
# PART 5: Save everything
# ══════════════════════════════════════════
artifact = {
    "model": best_model,
    "threshold": best_threshold,
    "best_params": best_params,
    "features": list(X_train.columns),
}
joblib.dump(artifact, MODEL_DIR / "model.joblib")
print(f"\nModel saved to {MODEL_DIR / 'model.joblib'}")

# Save confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, title in [
    (axes[0], y_pred_default, "Default Threshold (0.50)"),
    (axes[1], y_pred_opt, f"Optimized Threshold ({best_threshold:.3f})"),
]:
    cm = confusion_matrix(y_test, preds)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Failure"])
    ax.set_yticklabels(["Healthy", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=16, fontweight="bold")
plt.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORT_DIR / "08_confusion_matrices.png", dpi=150)
plt.close()
print("Confusion matrix plot saved.")

print("\n✅ Step 5 complete!")
