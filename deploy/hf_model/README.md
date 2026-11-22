---
license: mit
library_name: sklearn
tags:
- intrusion-detection
- tabular-classification
- scikit-learn
- nsl-kdd
metrics:
- f1
- accuracy
---

# HIDS - NSL-KDD intrusion classifier

A scikit-learn `Pipeline` (feature engineering -> imputation/scaling/one-hot -> Random Forest) that classifies a
host/connection record into **Normal, DoS, Probe, R2L (remote-to-local / password guessing), U2R (privilege escalation)**.
Built for the CT-378 Machine Learning course project; a rule-based layer maps each class to a simulated response action.

## Intended use
Educational demonstration of ML-based intrusion detection. **Not** for production security decisions.

## Training data
NSL-KDD (Tavallaee et al., 2009): `KDDTrain+` and `KDDTest+` combined, duplicates removed, 147,888 records,
41 features. Split 70/15/15 (stratified). Mirror used: https://github.com/defcom17/NSL_KDD

## Training procedure
Baseline (majority class) + Logistic Regression, Random Forest, Hist. Gradient Boosting, tuned with 5-fold GridSearchCV
(macro-F1, balanced class weights). Selected on the validation set. Best params: `{'max_depth': None, 'min_samples_leaf': 3}`.

## Results (held-out test set, evaluated once)
| Metric | Value |
|---|---|
| Accuracy | 0.9949 |
| Macro-F1 | 0.9205 |
| Weighted-F1 | 0.9949 |
| ROC-AUC (OvR, macro) | 0.9986 |

Per-class F1: Normal 0.995, DoS 0.999, Probe 0.993, R2L 0.949, U2R 0.667.

## Limitations
- NSL-KDD is an old benchmark (derived from 1998 traffic); real hosts look different.
- Random split of pooled data is optimistic: the official KDDTest+ contains attack types not seen in training and is harder.
- U2R is extremely rare (about 120 records), so its predictions are unreliable (test support: 18).
- The response actions are simulated.

## How to use
```python
import joblib, sys
sys.path.insert(0, ".")          # hids_core.py is included in this repo
import hids_core                 # needed to unpickle the feature-engineering step
model = joblib.load("hids_pipeline.joblib")
# df: DataFrame with the 41 NSL-KDD feature columns
print(model.predict(df))
```
Requires scikit-learn==1.9.1.
