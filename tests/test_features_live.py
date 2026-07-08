"""Feature engineering live test — validates all 4 feature modules."""
import sys, time
sys.path.insert(0, ".")

print("=" * 65)
print("  FEATURE ENGINEERING TEST — PHISHING DETECTOR")
print("=" * 65)

# ── Test 1: URL Features ──────────────────────────────────────────
print()
print("[1] URL Feature Extraction")
print("-" * 45)
from features.url_features import extract_url_features, get_feature_names

test_urls = [
    ("PHISHING", "http://paypal-secure-verify.account-confirm.xyz/login.php?token=abc&redirect=@185.23.1.4"),
    ("LEGIT",    "https://www.paypal.com/us/signin"),
    ("PHISHING", "http://192.168.1.10/amazon/update/billing.html"),
    ("LEGIT",    "https://amazon.com/s?k=laptop"),
]

for label, url in test_urls:
    f = extract_url_features(url)
    print(f"  [{label:8s}] {url[:62]}")
    print(
        f"           len={int(f['url_length']):3d}"
        f"  entropy={f['url_entropy']:.2f}"
        f"  subdomain={int(f['subdomain_depth'])}"
        f"  has_ip={int(f['has_ip'])}"
        f"  sus_kw={int(f['suspicious_kw_count'])}"
        f"  tld_risk={f['tld_risk_score']:.2f}"
        f"  brand_in_domain={int(f['brand_in_domain'])}"
    )

print()
print(f"  Total URL features: {len(get_feature_names())}")

# ── Test 2: NLP Features ─────────────────────────────────────────
print()
print("[2] NLP Feature Extraction")
print("-" * 45)
from features.nlp_features import extract_nlp_features

nlp_tests = [
    ("PHISHING", "http://paypal-secure-verify.xyz/update-account-now-immediately"),
    ("LEGIT",    "https://github.com/features/actions"),
]

for label, url in nlp_tests:
    f = extract_nlp_features(url)
    print(f"  [{label:8s}] {url[:55]}")
    print(
        f"           tokens={int(f['token_count'])}"
        f"  sus_density={f['suspicious_kw_density']:.3f}"
        f"  brand_score={int(f['brand_impersonation_score'])}"
        f"  phrase_hits={int(f['phishing_phrase_count'])}"
        f"  urgency={int(f['urgency_word_count'])}"
    )

# ── Test 3: Full Pipeline ─────────────────────────────────────────
print()
print("[3] Feature Pipeline — url_only mode (no network I/O)")
print("-" * 45)
from features.feature_pipeline import FeaturePipeline

pipeline = FeaturePipeline(mode="url_only")
print(f"  Pipeline feature dimensions: {pipeline.n_features}")

t0 = time.perf_counter()
result = pipeline.extract("http://paypal-account-verify.secure-login.xyz/update.php")
elapsed = (time.perf_counter() - t0) * 1000

print(f"  Extraction time : {elapsed:.2f}ms")
print(f"  Vector shape    : {result.vector.shape}")
print(f"  Vector dtype    : {result.vector.dtype}")
nonzero = int((result.vector != 0).sum())
print(f"  Non-zero features: {nonzero} / {len(result.vector)}")

# Top suspicious signals
top = sorted(result.raw_features.items(), key=lambda x: abs(float(x[1])), reverse=True)[:10]
print()
print("  Top 10 features by magnitude:")
for k, v in top:
    bar = "#" * min(int(float(v)), 30)
    print(f"    {k:<38s} = {float(v):>8.3f}  {bar}")

# ── Test 4: Batch extraction ──────────────────────────────────────
print()
print("[4] Batch Extraction (5 URLs)")
print("-" * 45)
urls = [
    "http://paypal-login.secure-update.xyz/verify.php",
    "https://paypal.com",
    "http://apple-id-verify.top/signin?user=abc",
    "https://apple.com/icloud",
    "http://secure-amazon-billing.info/update",
]
t0 = time.perf_counter()
results = pipeline.extract_batch(urls)
elapsed = (time.perf_counter() - t0) * 1000
for url, res in zip(urls, results):
    print(f"  {url[:55]:<55}  -> {res.vector.shape}  {res.extraction_time_ms:.1f}ms")
print(f"  Total batch time: {elapsed:.2f}ms  ({elapsed/len(urls):.1f}ms/URL)")

print()
print("=" * 65)
print("  All feature modules operational!")
print("=" * 65)
