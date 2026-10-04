"""Train, tune, evaluate and save the crop-suitability classifier.
Author: Harischandra Prasad Vissamsetti, M.C.A
Run from the project root:  python src/train.py
"""
import os, json, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             classification_report, roc_auc_score)
from sklearn.inspection import permutation_importance
import preprocessing
from preprocessing import FEATURES, TARGET, clean, load_data, make_pipeline

for _d in ("reports/figures", "models", "data"):
    os.makedirs(_d, exist_ok=True)          # recreate output folders if they were deleted

SEED, FIG = 42, "reports/figures/"
df = clean(load_data("data"))
X, y = df[FEATURES], df[TARGET]

# Split FIRST -> imputer/scaler are fitted on training data only (no leakage)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

candidates = {
    "Logistic Regression": (make_pipeline(LogisticRegression(max_iter=2000, random_state=SEED)),
                            {"model__C": [0.1, 1, 10, 100]}),
    "KNN": (make_pipeline(KNeighborsClassifier()),
            {"model__n_neighbors": [3, 5, 7, 11, 15], "model__weights": ["uniform", "distance"]}),
    "Decision Tree": (make_pipeline(DecisionTreeClassifier(random_state=SEED)),
                      {"model__max_depth": [4, 6, 8, 12, None], "model__min_samples_leaf": [1, 3, 5, 10]}),
}

results, fitted = {}, {}
for name, (pipe, grid) in candidates.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1).fit(Xtr, ytr)
    fitted[name] = gs.best_estimator_
    pred = gs.predict(Xte)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    auc = roc_auc_score(yte, gs.predict_proba(Xte), multi_class="ovr", average="macro", labels=gs.classes_)
    results[name] = dict(best_params={k.replace("model__", ""): v for k, v in gs.best_params_.items()},
                         cv_f1_macro=round(gs.best_score_, 4), test_accuracy=round(accuracy_score(yte, pred), 4),
                         test_precision=round(p, 4), test_recall=round(r, 4), test_f1=round(f, 4),
                         test_roc_auc=round(auc, 4))
    print(name, results[name])

# Model selection uses CROSS-VALIDATION scores only (never the test set).
# Models within 0.01 CV macro-F1 of the best are treated as comparable; the simplest is preferred
# (Logistic Regression > Decision Tree > KNN). It also gives smooth, informative probabilities.
order = ["Logistic Regression", "Decision Tree", "KNN"]
top_cv = max(results[n]["cv_f1_macro"] for n in order)
best = next(n for n in order if results[n]["cv_f1_macro"] >= top_cv - 0.01)
model = fitted[best]
pred = model.predict(Xte)
print("\nSELECTED:", best)

rep = classification_report(yte, pred, output_dict=True, zero_division=0)
open("reports/classification_report.txt", "w").write(classification_report(yte, pred, zero_division=0))

# ---- figures ----
pd.DataFrame(results).T[["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]] \
    .plot.bar(figsize=(9, 5), ylim=(0.8, 1.0))
plt.title("Model comparison (held-out test set)"); plt.xticks(rotation=0); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG + "08_model_comparison.png", dpi=130); plt.close()

labels = sorted(y.unique())
cm = confusion_matrix(yte, pred, labels=labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best}"); plt.tight_layout()
plt.savefig(FIG + "09_confusion_matrix.png", dpi=130); plt.close()

pi = permutation_importance(model, Xte, yte, n_repeats=15, random_state=SEED, scoring="f1_macro")
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
plt.figure(figsize=(7, 4)); imp.plot.barh(color="teal"); plt.title("Permutation importance (macro-F1 drop)")
plt.tight_layout(); plt.savefig(FIG + "10_feature_importance.png", dpi=130); plt.close()

# ---- error analysis ----
wrong = Xte.copy(); wrong["actual"], wrong["predicted"] = yte, pred
wrong = wrong[wrong.actual != wrong.predicted]
pairs = wrong.groupby(["actual", "predicted"]).size().sort_values(ascending=False).head(8)
per_class_f1 = pd.Series({l: rep[l]["f1-score"] for l in labels}).sort_values()
conf = model.predict_proba(Xte).max(axis=1)
correct = (pred == yte.values)
err = dict(n_test=len(yte), n_errors=int(len(wrong)),
           top_confusions={f"{a}->{b}": int(n) for (a, b), n in pairs.items()},
           weakest_classes={k: round(v, 3) for k, v in per_class_f1.head(5).items()},
           mean_confidence_correct=round(float(conf[correct].mean()), 3),
           mean_confidence_wrong=round(float(conf[~correct].mean()), 3) if (~correct).any() else None)
final_est = model.named_steps["model"]
cvscores = cross_val_score(make_pipeline(final_est.__class__(**final_est.get_params())),
                           X, y, cv=cv, scoring="accuracy")
err["cv_accuracy_mean_std_full_data"] = [round(cvscores.mean(), 4), round(cvscores.std(), 4)]
print(json.dumps(err, indent=2))

# ---- save model + metadata + metrics ----
meta = dict(selected_model=best, features=FEATURES, classes=list(model.classes_),
            feature_medians=X.median().round(2).to_dict(), train_rows=len(Xtr), test_rows=len(Xte),
            feature_ranges={c: [float(X[c].min()), float(X[c].max())] for c in FEATURES},
            dataset_file=preprocessing.LAST_SOURCE, n_records=int(len(df)), n_classes=int(y.nunique()))
joblib.dump(model, "models/crop_model.joblib")
json.dump(meta, open("models/metadata.json", "w"), indent=2)
json.dump(dict(results=results, selected=best, error_analysis=err, importance=imp.round(4).to_dict()),
          open("reports/metrics.json", "w"), indent=2)
print("\nSaved: models/crop_model.joblib, models/metadata.json, reports/metrics.json")
