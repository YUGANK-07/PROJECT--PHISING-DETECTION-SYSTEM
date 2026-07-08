"""Display training results summary."""
import json
from pathlib import Path

report_dir   = Path("reports")
artifact_dir = Path("models/artifacts")

print("=" * 62)
print("  FINAL MODEL COMPARISON -- PHISHING DETECTOR")
print("=" * 62)

models = [
    ("Random Forest", "rf"),
    ("XGBoost",       "xgb"),
    ("Ensemble",      "ensemble"),
]

header = f"  {'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}"
print(header)
print("  " + "-" * 56)

for name, key in models:
    path = report_dir / f"{key}_evaluation.json"
    if path.exists():
        r = json.loads(path.read_text())
        print(
            f"  {name:<20}"
            f" {r['precision']:>10.4f}"
            f" {r['recall']:>10.4f}"
            f" {r['f1']:>10.4f}"
            f" {r['roc_auc']:>10.4f}"
        )

print("=" * 62)
print()

# Confusion matrix summary
print("  Confusion Matrix Details (Ensemble on 37,500 test URLs):")
ens_path = report_dir / "ensemble_evaluation.json"
if ens_path.exists():
    r = json.loads(ens_path.read_text())
    print(f"    True Legit  -> Legit  (TN): {r['tn']:>8,}")
    print(f"    True Legit  -> Phish  (FP): {r['fp']:>8,}  (false alarms)")
    print(f"    True Phish  -> Legit  (FN): {r['fn']:>8,}  (MISSED phishing!)")
    print(f"    True Phish  -> Phish  (TP): {r['tp']:>8,}")
    print(f"    FPR (false alarm rate)     : {r['fpr']:.4f}")
    print(f"    FNR (missed phishing rate) : {r['fnr']:.4f}")

print()

# Artifacts
print("  Saved model artifacts:")
for f in sorted(artifact_dir.rglob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        rel = str(f).replace(str(artifact_dir) + "\\", "")
        print(f"    {rel:<40}  {size_kb:>8.1f} KB")

print()
print("  Report plots generated:")
for f in sorted(report_dir.glob("*.png")):
    print(f"    {f.name}")

print()
print("  Training complete! Models ready for API deployment.")
