"""Quick sanity check on real-world URLs."""
import sys; sys.path.insert(0, ".")
from features.feature_pipeline import FeaturePipeline
import joblib, numpy as np
from pathlib import Path

art  = Path("models/artifacts")
from models.random_forest import PhishingRandomForest
from models.xgboost_model import PhishingXGBoost
rf   = PhishingRandomForest.load(art / "random_forest.joblib")
xgb  = PhishingXGBoost.load(art / "xgboost.joblib")
meta = joblib.load(art / "meta_learner.joblib")
pipe = FeaturePipeline(mode="url_only")

tests = [
    ("LEGIT",    "https://www.paypal.com/signin"),
    ("LEGIT",    "https://amazon.com/s?k=laptop"),
    ("LEGIT",    "https://github.com/features"),
    ("LEGIT",    "https://google.com"),
    ("LEGIT",    "https://chase.com/personal/checking"),
    ("PHISHING", "http://paypal-secure-verify.account-confirm.xyz/login.php"),
    ("PHISHING", "http://apple-id-verify.secure-update.top/signin"),
    ("PHISHING", "http://192.168.1.5/amazon/billing.php"),
    ("PHISHING", "http://microsoft-account-suspend.xyz/verify.php"),
    ("PHISHING", "http://secure-amazon-billing.info/update.php?token=abc123"),
]

print()
print(f"  {'Expected':<10}  {'Score':>7}  {'Verdict':<8}  URL")
print("  " + "-"*70)
correct = 0
for label, url in tests:
    r    = pipe.extract(url)
    X    = r.vector.reshape(1, -1)
    base = np.column_stack([rf.predict_proba(X), xgb.predict_proba(X)])
    ens  = float(meta.predict_proba(base)[0, 1])
    pred = "PHISHING" if ens > 0.5 else "LEGIT"
    ok   = pred == label
    mark = "✓" if ok else "✗"
    correct += int(ok)
    risk = "High" if ens > 0.7 else "Medium" if ens > 0.4 else "Low"
    print(f"  {label:<10}  {ens*100:>6.1f}%  {risk:<8}  {mark}  {url[:55]}")

print()
print(f"  Accuracy: {correct}/{len(tests)} correct ({100*correct//len(tests)}%)")
