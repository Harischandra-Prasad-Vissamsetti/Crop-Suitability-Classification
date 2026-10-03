"""Exploratory data analysis -> reports/figures/*.png and reports/eda_summary.txt"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from preprocessing import FEATURES, TARGET, VALID_RANGES, clean, load_data

FIG = "reports/figures/"
raw = load_data("data")


out = []
out.append(f"Raw shape: {raw.shape}")
out.append(f"Duplicate rows: {raw.duplicated().sum()}")
out.append("Missing values per column:\n" + raw.isna().sum().to_string())
bad = {c: int(((raw[c] < lo) | (raw[c] > hi)).sum()) for c, (lo, hi) in VALID_RANGES.items()}
out.append(f"Out-of-range (impossible) values: {bad}")
out.append("Describe (raw):\n" + raw[FEATURES].describe().round(2).to_string())
df = clean(raw)
out.append(f"\nAfter cleaning shape: {df.shape}")
vc = df[TARGET].value_counts()
out.append(f"Class balance: min={vc.min()}, max={vc.max()}, n_classes={len(vc)}")

# 1 class balance
plt.figure(figsize=(9,5)); vc.sort_values().plot.barh(color="seagreen")
plt.title("Class distribution (after cleaning)"); plt.xlabel("Samples"); plt.tight_layout()
plt.savefig(FIG+"01_class_balance.png", dpi=130); plt.close()
# 2 missing values
plt.figure(figsize=(7,4)); df[FEATURES].isna().sum().plot.bar(color="indianred")
plt.title("Missing values (after cleaning, incl. nulled outliers)"); plt.tight_layout()
plt.savefig(FIG+"02_missing.png", dpi=130); plt.close()
# 3 distributions
fig, ax = plt.subplots(2, 4, figsize=(16,7))
for a, c in zip(ax.ravel(), FEATURES):
    sns.histplot(df[c], kde=True, ax=a, color="steelblue"); a.set_title(c)
ax.ravel()[-1].axis("off"); plt.tight_layout(); plt.savefig(FIG+"03_distributions.png", dpi=120); plt.close()
# 4 outliers in RAW data
fig, ax = plt.subplots(1, 7, figsize=(18,4))
for a, c in zip(ax, FEATURES): sns.boxplot(y=raw[c], ax=a, color="orange"); a.set_title(c)
plt.suptitle("Raw data boxplots - outliers visible"); plt.tight_layout()
plt.savefig(FIG+"04_outliers_raw.png", dpi=120); plt.close()
# 5 correlation
plt.figure(figsize=(7,6)); sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature correlation"); plt.tight_layout(); plt.savefig(FIG+"05_correlation.png", dpi=130); plt.close()
hi = df[FEATURES].corr().abs().where(~np.eye(7, dtype=bool)).stack()
out.append("Max |corr| between features: %s = %.2f" % (hi.idxmax(), hi.max()))
# 6 feature by crop (mean heatmap, z-scored)
m = df.groupby(TARGET)[FEATURES].mean()
plt.figure(figsize=(9,8)); sns.heatmap((m-m.mean())/m.std(), cmap="RdYlGn", annot=m.round(0), fmt="g", cbar_kws={"label":"z-score"})
plt.title("Mean feature value per crop (annotated raw means)"); plt.tight_layout()
plt.savefig(FIG+"06_crop_profiles.png", dpi=130); plt.close()
# 7 key features per crop
fig, ax = plt.subplots(1, 3, figsize=(18,6))
for a, c in zip(ax, ["rainfall","temperature","humidity"]):
    order = df.groupby(TARGET)[c].median().sort_values().index
    sns.boxplot(data=df, y=TARGET, x=c, order=order, ax=a, color="lightgreen", fliersize=2)
plt.tight_layout(); plt.savefig(FIG+"07_features_by_crop.png", dpi=120); plt.close()
open("reports/eda_summary.txt","w").write("\n".join(out)); print("\n".join(out))