"""Diagnose typosquat feature gaps."""
import sys; sys.path.insert(0, ".")
from features.feature_pipeline import FeaturePipeline
import joblib, numpy as np
from pathlib import Path

art = Path("models/artifacts")
from models.random_forest import PhishingRandomForest
from models.xgboost_model import PhishingXGBoost
rf   = PhishingRandomForest.load(art / "random_forest.joblib")
xgb  = PhishingXGBoost.load(art / "xgboost.joblib")
meta = joblib.load(art / "meta_learner.joblib")
pipe = FeaturePipeline(mode="url_only")

tests = [
    ("LEGIT",    "https://microsoft.com"),
    ("PHISHING", "https://rnicrosoft.com"),
    ("PHISHING", "https://micros0ft.com"),
    ("LEGIT",    "https://paypal.com"),
    ("PHISHING", "https://paypa1.com"),
    ("PHISHING", "https://paypai.com"),
    ("LEGIT",    "https://google.com"),
    ("PHISHING", "https://g00gle.com"),
    ("LEGIT",    "https://amazon.com"),
    ("PHISHING", "https://arnazon.com"),
    ("PHISHING", "https://amaz0n.com"),
    ("PHISHING", "https://appleid-verify.com/login"),
    ("PHISHING", "https://secure.paypal-update.com"),
]

print(f"\n  {'Expected':<10}  {'Score':>7}  {'brand':>6}  {'entropy':>8}  {'sus_kw':>7}  URL")
print("  " + "-"*80)
for label, url in tests:
    r    = pipe.extract(url)
    X    = r.vector.reshape(1, -1)
    base = np.column_stack([rf.predict_proba(X), xgb.predict_proba(X)])
    ens  = float(meta.predict_proba(base)[0, 1])
    f    = r.raw_features
    ok   = (ens > 0.5) == (label == "PHISHING")
    mark = "OK" if ok else "WRONG"
    print(
        f"  {label:<10}  {ens*100:>6.1f}%"
        f"  {int(f.get('brand_in_domain',0)):>6}"
        f"  {f.get('url_entropy',0):>8.3f}"
        f"  {int(f.get('suspicious_kw_count',0)):>7}"
        f"  [{mark}]  {url}"
    )
