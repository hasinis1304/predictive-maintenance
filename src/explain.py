import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, SEED, TEST_SIZE, TARGET, MODEL_DIR, REPORT_DIR
from src.features import get_X_y

REPORT_DIR.mkdir(exist_ok=True)

artifact = joblib.load(MODEL_DIR / "model.joblib")
model = artifact["model"]
threshold = artifact["threshold"]

df = pd.read_csv(DATA_PATH)
X, y = get_X_y(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

print(f"Model loaded. Threshold: {threshold:.3f}")
print(f"Computing SHAP values for {X_test.shape[0]} test samples...\n")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values = shap_values[1]
ev = explainer.expected_value
if hasattr(ev, "__len__"):
    ev = float(ev[0]) if len(ev) == 1 else float(ev[1])
else:
    ev = float(ev)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance", fontsize=13)
plt.tight_layout()
plt.savefig(REPORT_DIR / "09_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("1/4 SHAP summary plot saved")

fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Mean SHAP Value by Feature", fontsize=13)
plt.tight_layout()
plt.savefig(REPORT_DIR / "10_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("2/4 SHAP bar plot saved")

test_probs = model.predict_proba(X_test)[:, 1]
y_pred = (test_probs >= threshold).astype(int)
features = list(X_test.columns)


def make_waterfall(idx, title, filename):
    vals = shap_values[idx]
    data = X_test.iloc[idx].values
    order = np.argsort(np.abs(vals))
    sorted_vals = vals[order]
    sorted_names = [features[i] + " = " + str(round(data[i], 1)) for i in order]
    cumulative = ev + np.cumsum(np.concatenate([[0], sorted_vals[:-1]]))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in sorted_vals]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(sorted_vals)), sorted_vals, left=cumulative, color=colors)
    ax.set_yticks(range(len(sorted_vals)))
    ax.set_yticklabels(sorted_names)
    ax.axvline(x=ev, color="gray", linestyle="--", linewidth=1, label="Base: " + str(round(ev, 3)))
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


tp_mask = (y_test.values == 1) & (y_pred == 1)
if tp_mask.sum() > 0:
    tp_idx = np.where(tp_mask)[0][0]
    prob_str = str(round(test_probs[tp_idx] * 100, 1)) + "%"
    make_waterfall(tp_idx, "Why This Machine Was Flagged (prob=" + prob_str + ")", "11_shap_failure_waterfall.png")
    print("3/4 Failure waterfall saved")
    print("\n   Sample #" + str(tp_idx) + ": predicted prob = " + prob_str)
    for i, feat in enumerate(features):
        val = round(X_test.iloc[tp_idx][feat], 2)
        sv = round(shap_values[tp_idx][i], 3)
        print("   " + feat.rjust(18) + " = " + str(val).rjust(10) + "  (SHAP = " + str(sv) + ")")
else:
    print("No true positives found")

tn_mask = (y_test.values == 0) & (y_pred == 0)
if tn_mask.sum() > 0:
    tn_idx = np.where(tn_mask)[0][0]
    prob_str = str(round(test_probs[tn_idx] * 100, 1)) + "%"
    make_waterfall(tn_idx, "Why This Machine Was Cleared (prob=" + prob_str + ")", "12_shap_healthy_waterfall.png")
    print("4/4 Healthy waterfall saved")

importance = pd.DataFrame({
    "Feature": features,
    "Mean |SHAP|": np.abs(shap_values).mean(axis=0)
}).sort_values("Mean |SHAP|", ascending=False)

print("\n" + "=" * 40)
print("FEATURE IMPORTANCE RANKING")
print("=" * 40)
print(importance.to_string(index=False))
print("\nAll plots saved to " + str(REPORT_DIR.resolve()))
print("\nStep 6 complete!")