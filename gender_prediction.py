"""
Department-store gender prediction — final solution (ROC-AUC).
Goal: robust PRIVATE-leaderboard performance, not Public overfitting.
Final features: perk signal (Discount/Promo/Subscription) + holdout-validated
pay_matches_pref.  CV ROC-AUC = 0.8194.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub   = pd.read_csv("sample_submission.csv")
y = (train["Gender"] == "Male").astype(int).values

def signal_features(df):
    disc = (df["Discount Applied"]    == "Yes").astype(int)
    sub_ = (df["Subscription Status"] == "Yes").astype(int)
    return pd.DataFrame({
        "disc": disc,
        "promo": (df["Promo Code Used"] == "Yes").astype(int),
        "sub":  sub_,
        "any_perk": ((disc + sub_) > 0).astype(int),
        # holdout-validated: +0.005 AUC, win% 68-88 across 7 seed sets
        "pay_matches_pref": (df["Payment Method"] == df["Preferred Payment Method"]).astype(int),
    })

Xtr, Xte = signal_features(train), signal_features(test)

# Cross-validated AUC
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cvs = cross_val_score(model, Xtr, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"CV ROC-AUC: {cvs.mean():.4f} +/- {cvs.std():.4f}")   # 0.8194

# Seed-averaged ensemble -> predict -> submission
proba = np.zeros(len(test))
for s in [0, 1, 7, 21, 42, 101, 202, 777, 1234, 2024]:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
    m.fit(Xtr, y)
    proba += m.predict_proba(Xte)[:, 1]
proba /= 10

out = sub[["id"]].merge(pd.DataFrame({"id": test["id"], "Male_prob": proba}), on="id")
assert out["Male_prob"].between(0, 1).all() and len(out) == len(sub)
out.to_csv("submission.csv", index=False)
