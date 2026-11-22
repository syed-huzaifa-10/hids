# %% [markdown]
# # Host-Based Intrusion Detection & Automated Response System (HIDS)
# **CT-378 Machine Learning - semester project.**
#
# Goal: classify each host/connection record into one of five classes
# (Normal, DoS, Probe, R2L, U2R) and map the predicted class to an automated
# response action.
#
# **Data:** NSL-KDD (Tavallaee et al., 2009). Two public files are combined:
# `KDDTrain+.txt` and `KDDTest+.txt`. Source mirror:
# https://github.com/defcom17/NSL_KDD . See `data/README.md`.
#
# **Target:** `target` (5 classes). **Success metric:** macro-F1 (the data is
# heavily imbalanced, so plain accuracy is misleading).

# %%
import sys, json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, sklearn

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from hids_core import *   # schema, add_features, response engine

warnings.filterwarnings("ignore")
RNG = 42
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

# colour per class (categorical palette slots 1,2,3,7,8; fixed order)
CCOL = {"Normal": "#2a78d6", "DoS": "#eb6834", "Probe": "#1baf7a",
        "R2L": "#6250d6", "U2R": "#e34948"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e5e2", "grid.linewidth": 0.6,
                     "axes.axisbelow": True})

# %% [markdown]
# ## 1. Load and combine the two sources

# %%
a = load_raw(ROOT / "data" / "KDDTrain+.txt"); a["source"] = "KDDTrain+"
b = load_raw(ROOT / "data" / "KDDTest+.txt");  b["source"] = "KDDTest+"
raw = pd.concat([a, b], ignore_index=True)
print("rows before de-duplication:", len(raw))
dups = raw.duplicated(subset=FEATURES + ["target"]).sum()
df = raw.drop_duplicates(subset=FEATURES + ["target"]).reset_index(drop=True)
print("duplicates removed:", dups, "-> rows:", len(df), "| features:", len(FEATURES))
print("missing values:", int(df[FEATURES].isna().sum().sum()))
df["target"].value_counts()

# %% [markdown]
# ## 2. Exploratory data analysis (6 plots)

# %%
# Plot 1 - class balance
vc = df["target"].value_counts().reindex(CLASSES)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.bar(vc.index, vc.values, color=[CCOL[c] for c in vc.index], width=0.6)
for i, v in enumerate(vc.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
ax.set_ylabel("records"); ax.set_title("Class balance (5 classes)")
fig.tight_layout(); fig.savefig(FIG / "eda_class_balance.png"); plt.show()
print((vc / vc.sum() * 100).round(2).to_dict())

# %%
# Plot 2 - protocol type by class (share within class)
ct = pd.crosstab(df["target"], df["protocol_type"], normalize="index").reindex(CLASSES) * 100
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ct.plot(kind="bar", stacked=True, ax=ax, color=["#2a78d6", "#eda100", "#1baf7a"],
        width=0.6, edgecolor="white", linewidth=1)
ax.set_ylabel("% of class"); ax.set_xlabel(""); ax.set_title("Protocol by class")
ax.legend(title="", fontsize=7, frameon=False); plt.xticks(rotation=0)
fig.tight_layout(); fig.savefig(FIG / "eda_protocol.png"); plt.show()

# %%
# Plot 3 - outliers: log(src_bytes) per class (boxplot)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
data = [np.log1p(df.loc[df.target == c, "src_bytes"]) for c in CLASSES]
bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True, widths=0.5,
                flierprops=dict(marker=".", markersize=2, alpha=.3))
for p, c in zip(bp["boxes"], CLASSES):
    p.set_facecolor(CCOL[c]); p.set_alpha(.75)
ax.set_ylabel("log(1 + src_bytes)"); ax.set_title("Outliers: bytes sent per class")
fig.tight_layout(); fig.savefig(FIG / "eda_bytes_boxplot.png"); plt.show()

# %%
# Plot 4 - error rates by class (mean)
cols = ["serror_rate", "rerror_rate", "dst_host_serror_rate"]
m = df.groupby("target")[cols].mean().reindex(CLASSES)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
m.plot(kind="bar", ax=ax, color=["#2a78d6", "#eb6834", "#1baf7a"], width=0.75,
       edgecolor="white", linewidth=1)
ax.set_ylabel("mean rate"); ax.set_xlabel(""); ax.set_title("Connection error rates")
ax.legend(fontsize=6.5, frameon=False); plt.xticks(rotation=0)
fig.tight_layout(); fig.savefig(FIG / "eda_error_rates.png"); plt.show()

# %%
# Plot 5 - host-level privilege indicators by class (R2L / U2R signals)
cols = ["num_failed_logins", "root_shell", "su_attempted", "num_root", "num_shells", "num_file_creations"]
m = df.groupby("target")[cols].mean().reindex(CLASSES)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
im = ax.imshow(m.values.T, aspect="auto", cmap="Blues")
ax.set_xticks(range(5)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(len(cols)))
ax.set_yticklabels(cols, fontsize=7); ax.grid(False)
for i in range(5):
    for j in range(len(cols)):
        v = m.values[i, j]; ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=6,
                                      color="white" if v > m.values.max() * .55 else "black")
ax.set_title("Host-level indicators (class mean)")
fig.tight_layout(); fig.savefig(FIG / "eda_host_indicators.png"); plt.show()

# %%
# Plot 6 - correlation heatmap of numeric features
num = df[NUMERIC].loc[:, df[NUMERIC].std() > 0]
corr = num.corr()
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Correlation of {corr.shape[0]} numeric features")
fig.colorbar(im, ax=ax, fraction=0.04)
fig.tight_layout(); fig.savefig(FIG / "eda_correlation.png"); plt.show()
hi = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
print("pairs with |r| > 0.9:", int((hi.abs() > 0.9).sum()))
print("constant columns dropped from the plot:", [c for c in NUMERIC if df[c].std() == 0])

# %% [markdown]
# **EDA findings.** DoS and Normal dominate; U2R is extremely rare (<0.1%), so
# macro-F1 and class weighting are needed. DoS shows very high SYN-error rates,
# R2L shows failed logins / guest logins, U2R shows root-shell and `su` /
# `num_root` signals. Many `*_rate` columns are strongly correlated.
# `src_bytes` is heavy-tailed, hence the log transform in feature engineering.
# There are no missing values (imputers are kept so the deployed pipeline is
# robust to blank inputs).

# %% [markdown]
# ## 3. Train / validation / test split (70 / 15 / 15, stratified)

# %%
from sklearn.model_selection import train_test_split
X, y = df[FEATURES], df["target"]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RNG)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RNG)
print(len(X_tr), len(X_val), len(X_te))
pd.DataFrame({"train": y_tr.value_counts(), "val": y_val.value_counts(),
              "test": y_te.value_counts()}).reindex(CLASSES)

# %% [markdown]
# ## 4. Pre-processing pipeline (fitted on training data only)
# Feature engineering (5 new columns) -> impute -> scale numerics / one-hot
# categoricals -> classifier. Everything lives inside one scikit-learn `Pipeline`.

# %%
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def make_pipeline(clf):
    num_cols = NUMERIC + ENGINEERED
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL),
    ])
    return Pipeline([("fe", FunctionTransformer(add_features, validate=False)),
                     ("pre", pre), ("clf", clf)])

# %% [markdown]
# ## 5. Baseline + three algorithms, tuned with 5-fold GridSearchCV
# * Baseline: `DummyClassifier` (always predicts the majority class)
# * Logistic Regression, Random Forest, Histogram Gradient Boosting
# * Scoring: macro-F1; class weights are balanced to handle the imbalance.

# %%
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
specs = {
    "Logistic Regression": (LogisticRegression(max_iter=300, class_weight="balanced"),
                            {"clf__C": [0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                             n_jobs=2, random_state=RNG),
                      {"clf__max_depth": [15, None], "clf__min_samples_leaf": [1, 3]}),
    "Hist. Gradient Boosting": (HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                                               early_stopping=True, random_state=RNG),
                                {"clf__learning_rate": [0.05, 0.15], "clf__max_leaf_nodes": [15, 31]}),
}
rows, tuned = [], {}

base = make_pipeline(DummyClassifier(strategy="most_frequent")).fit(X_tr, y_tr)
rows.append({"Model": "Baseline (majority class)", "Best params": "-", "CV macro-F1": np.nan,
             "Train macro-F1": f1_score(y_tr, base.predict(X_tr), average="macro"),
             "Val macro-F1": f1_score(y_val, base.predict(X_val), average="macro"),
             "Val accuracy": accuracy_score(y_val, base.predict(X_val)), "Fit time (s)": 0})

for name, (clf, grid) in specs.items():
    t0 = time.time()
    gs = GridSearchCV(make_pipeline(clf), grid, cv=cv, scoring="f1_macro", n_jobs=1, refit=True)
    gs.fit(X_tr, y_tr)
    tuned[name] = gs.best_estimator_
    pv = gs.predict(X_val)
    rows.append({"Model": name, "Best params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                 "CV macro-F1": gs.best_score_,
                 "Train macro-F1": f1_score(y_tr, gs.predict(X_tr), average="macro"),
                 "Val macro-F1": f1_score(y_val, pv, average="macro"),
                 "Val accuracy": accuracy_score(y_val, pv), "Fit time (s)": time.time() - t0})
    print(name, "done in", round(time.time() - t0), "s")

results = pd.DataFrame(rows)
results.round(4)

# %% [markdown]
# ## 6. Model selection on the validation set

# %%
best_name = results.iloc[1:].sort_values("Val macro-F1", ascending=False).iloc[0]["Model"]
print("selected:", best_name)
results.to_csv(ROOT / "models" / "model_comparison.csv", index=False)
fig, ax = plt.subplots(figsize=(4.6, 2.6))
r = results.copy()
x = np.arange(len(r)); w = 0.38
ax.bar(x - w/2, r["Train macro-F1"], w, label="train", color="#2a78d6")
ax.bar(x + w/2, r["Val macro-F1"], w, label="validation", color="#eb6834")
ax.set_xticks(x); ax.set_xticklabels(["Baseline", "LogReg", "RandForest", "HistGB"], fontsize=7)
ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=7)
ax.set_title("Train vs validation (over/under-fitting check)")
fig.tight_layout(); fig.savefig(FIG / "model_comparison.png"); plt.show()

# %% [markdown]
# ## 7. Final evaluation - held-out test set, used once

# %%
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             ConfusionMatrixDisplay)
final = tuned[best_name]
pred = final.predict(X_te)
proba = final.predict_proba(X_te)
test_macro = f1_score(y_te, pred, average="macro")
test_weighted = f1_score(y_te, pred, average="weighted")
test_acc = accuracy_score(y_te, pred)
test_auc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=list(final.classes_))
print(f"TEST  accuracy={test_acc:.4f}  macro-F1={test_macro:.4f}  weighted-F1={test_weighted:.4f}  ROC-AUC(ovr)={test_auc:.4f}")
rep = classification_report(y_te, pred, labels=CLASSES, output_dict=True, zero_division=0)
print(classification_report(y_te, pred, labels=CLASSES, digits=4, zero_division=0))

cm = confusion_matrix(y_te, pred, labels=CLASSES)
fig, ax = plt.subplots(figsize=(4.0, 3.4))
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.grid(False); ax.set_title(f"Test confusion matrix - {best_name}")
fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png"); plt.show()

# %% [markdown]
# ### Error analysis

# %%
err = pd.DataFrame({"true": y_te.values, "pred": pred})
err = err[err.true != err.pred].value_counts().reset_index(name="n").head(8)
print("total test errors:", int((y_te.values != pred).sum()), "of", len(y_te))
err

# %% [markdown]
# ## 8. Which features matter? (permutation importance on validation data)

# %%
from sklearn.inspection import permutation_importance
sub = X_val.sample(5000, random_state=RNG); ysub = y_val.loc[sub.index]
pi = permutation_importance(final, sub, ysub, scoring="f1_macro", n_repeats=3,
                            random_state=RNG, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
imp.rename("importance").to_csv(ROOT / "models" / "feature_importance.csv")
top = imp.head(12)[::-1]
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.barh(top.index, top.values, color="#2a78d6", height=0.6)
ax.set_xlabel("drop in macro-F1 when shuffled"); ax.set_title("Top-12 features (permutation)")
fig.tight_layout(); fig.savefig(FIG / "feature_importance.png"); plt.show()
imp.head(12)

# %% [markdown]
# ## 9. Save the final pipeline (joblib) and metadata for the apps

# %%
joblib.dump(final, ROOT / "models" / "hids_pipeline.joblib", compress=3)

top_features = list(imp.head(12).index)
defaults = {}
for c in FEATURES:
    defaults[c] = (X_tr[c].mode().iloc[0] if c in CATEGORICAL else float(X_tr[c].median()))
ranges = {c: [float(X_tr[c].min()), float(X_tr[c].max())] for c in NUMERIC}

# demo events: 4 correctly-classified test records per class (kept in the repo for the apps)
ev = X_te.copy(); ev["true_class"] = y_te.values; ev["pred"] = pred
ev = ev[ev.true_class == ev.pred]
ev = pd.concat([ev[ev.true_class == c].sample(4, random_state=RNG) for c in CLASSES])
ev = ev.drop(columns="pred").reset_index(drop=True)
ev.to_csv(ROOT / "app" / "sample_events.csv", index=False)

meta = {
    "classes": list(final.classes_), "best_model": best_name,
    "best_params": results.set_index("Model").loc[best_name, "Best params"],
    "test": {"accuracy": test_acc, "macro_f1": test_macro, "weighted_f1": test_weighted,
             "roc_auc_ovr_macro": test_auc,
             "per_class_f1": {c: rep[c]["f1-score"] for c in CLASSES}},
    "top_features": top_features, "defaults": defaults, "ranges": ranges,
    "services": sorted(df["service"].unique()), "protocols": sorted(df["protocol_type"].unique()),
    "flags": sorted(df["flag"].unique()),
    "n_rows": int(len(df)), "sklearn_version": sklearn.__version__,
}
json.dump(meta, open(ROOT / "models" / "metadata.json", "w"), indent=2, default=str)

# sanity check: reload and re-predict
reloaded = joblib.load(ROOT / "models" / "hids_pipeline.joblib")
assert (reloaded.predict(X_te.head(200)) == final.predict(X_te.head(200))).all()
print("saved. sklearn", sklearn.__version__)
