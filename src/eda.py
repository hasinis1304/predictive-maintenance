import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_PATH, TARGET, FAILURE_MODES, REPORT_DIR

REPORT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

df = pd.read_csv(DATA_PATH)

# ── 1. Class imbalance ──
fig, ax = plt.subplots(figsize=(6, 4))
counts = df[TARGET].value_counts()
bars = ax.bar(["Healthy (0)", "Failure (1)"], counts.values,
              color=["#2ecc71", "#e74c3c"])
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
            f"{val}\n({val/len(df):.1%})", ha="center", fontweight="bold")
ax.set_ylabel("Count")
ax.set_title("Target Distribution — Severe Class Imbalance")
plt.tight_layout()
plt.savefig(REPORT_DIR / "01_class_imbalance.png", dpi=150)
plt.close()
print("✅ 1/6 Class imbalance plot saved")

# ── 2. Failure mode breakdown ──
fig, ax = plt.subplots(figsize=(7, 4))
mode_counts = df[FAILURE_MODES].sum().sort_values(ascending=True)
colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
bars = ax.barh(mode_counts.index, mode_counts.values, color=colors)
for bar, val in zip(bars, mode_counts.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontweight="bold")
ax.set_xlabel("Count")
ax.set_title("Failure Mode Distribution")
plt.tight_layout()
plt.savefig(REPORT_DIR / "02_failure_modes.png", dpi=150)
plt.close()
print("✅ 2/6 Failure modes plot saved")

# ── 3. Sensor distributions: healthy vs failed ──
sensor_cols = ["Air temperature [K]", "Process temperature [K]",
               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, col in enumerate(sensor_cols):
    ax = axes[i]
    for label, color, name in [(0, "#2ecc71", "Healthy"), (1, "#e74c3c", "Failure")]:
        subset = df[df[TARGET] == label][col]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=name, density=True)
    ax.set_title(col)
    ax.legend()
axes[5].axis("off")
plt.suptitle("Sensor Distributions: Healthy vs Failure", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORT_DIR / "03_sensor_distributions.png", dpi=150)
plt.close()
print("✅ 3/6 Sensor distributions saved")

# ── 4. Boxplots by failure status ──
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, col in enumerate(sensor_cols):
    sns.boxplot(data=df, x=TARGET, y=col, ax=axes[i],
                palette={0: "#2ecc71", 1: "#e74c3c"}, hue=TARGET, legend=False)
    axes[i].set_xticklabels(["Healthy", "Failure"])
    axes[i].set_xlabel("")
plt.suptitle("Sensor Values by Machine Status", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORT_DIR / "04_boxplots.png", dpi=150)
plt.close()
print("✅ 4/6 Boxplots saved")

# ── 5. Correlation heatmap ──
numeric_cols = sensor_cols + [TARGET]
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "05_correlation.png", dpi=150)
plt.close()
print("✅ 5/6 Correlation heatmap saved")

# ── 6. Failure rate by machine type ──
fig, ax = plt.subplots(figsize=(6, 4))
type_fail = df.groupby("Type")[TARGET].mean().reindex(["L", "M", "H"]) * 100
bars = ax.bar(type_fail.index, type_fail.values, color=["#3498db", "#f39c12", "#e74c3c"])
for bar, val in zip(bars, type_fail.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Failure Rate (%)")
ax.set_title("Failure Rate by Machine Type (L=Low, M=Medium, H=High Quality)")
plt.tight_layout()
plt.savefig(REPORT_DIR / "06_failure_by_type.png", dpi=150)
plt.close()
print("✅ 6/6 Failure rate by type saved")

# ── Summary statistics for report ──
print("\n" + "=" * 50)
print("KEY FINDINGS FOR YOUR REPORT")
print("=" * 50)
print(f"Total samples:        {len(df)}")
print(f"Failure rate:         {df[TARGET].mean():.2%}")
print(f"Imbalance ratio:      1:{int(counts[0]/counts[1])} (failure:healthy)")
print(f"Most common failure:  HDF (Heat Dissipation Failure) — {df['HDF'].sum()} cases")
print(f"Rarest failure:       RNF (Random Failure) — {df['RNF'].sum()} cases")
print(f"Temp correlation:     {corr.loc['Air temperature [K]','Process temperature [K]']:.2f}")
print(f"Torque-RPM corr:      {corr.loc['Torque [Nm]','Rotational speed [rpm]']:.2f}")
print(f"L-type failure rate:  {type_fail['L']:.1f}%")
print(f"H-type failure rate:  {type_fail['H']:.1f}%")
print(f"\nAll plots saved to: {REPORT_DIR.resolve()}")
