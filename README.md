# PhishGuard — Production-Grade Phishing Detection System

A scalable, explainable, real-time phishing detection system built with Machine Learning, FastAPI, and a premium web UI.

---

## 🚀 Quick Start

### 1. Start the API server
```powershell
cd C:\Users\yash0\Desktop\PROJECT\phishing-detector
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2. Open the Web UI
```
http://localhost:8000/ui
```

### 3. API Documentation (Swagger)
```
http://localhost:8000/docs
```

---

## 🏗️ Architecture

```
phishing-detector/
├── data/
│   ├── scripts/              # PhishTank, OpenPhish, Tranco fetchers
│   ├── preprocess.py         # Cleaning, dedup, balancing, splitting
│   └── pipeline.py           # One-command data pipeline CLI
│
├── features/
│   ├── url_features.py       # 45 lexical URL features (entropy, TLD risk, homographs…)
│   ├── domain_features.py    # 19 WHOIS / DNS / SSL features
│   ├── webpage_features.py   # 31 HTML / JS structural features
│   ├── nlp_features.py       # 7 NLP features + optional DistilBERT (768-dim)
│   └── feature_pipeline.py   # Unified extractor → 102-dim float32 vector
│
├── models/
│   ├── random_forest.py      # sklearn RandomForestClassifier wrapper
│   ├── xgboost_model.py      # XGBoost + early stopping + Optuna tuning
│   ├── neural_network.py     # PyTorch MLP with BatchNorm, GELU, cosine LR
│   ├── ensemble.py           # Stacking ensemble (OOF + LogisticRegression meta)
│   ├── evaluator.py          # Metrics, ROC/PR curves, confusion matrix plots
│   ├── explainer.py          # SHAP TreeExplainer + human-readable summaries
│   ├── trainer.py            # End-to-end training CLI
│   └── artifacts/            # Saved model files (.joblib, .pt)
│
├── api/
│   ├── main.py               # FastAPI app factory + lifespan model loading
│   ├── schemas.py            # Pydantic v2 request/response models
│   ├── cache.py              # Async Redis cache layer (graceful degradation)
│   ├── security.py           # JWT + API key auth
│   └── routers/
│       ├── predict.py        # POST /predict, POST /predict/batch
│       └── health.py         # GET /health, GET /metrics
│
├── frontend/
│   ├── index.html            # Single-page application
│   ├── style.css             # Premium dark glassmorphism UI
│   └── app.js                # API client + rendering + history
│
├── utils/
│   ├── config.py             # Pydantic settings + .env
│   ├── logger.py             # Loguru structured logging
│   └── helpers.py            # URL normalization, entropy, domain utils
│
└── reports/                  # Evaluation plots (ROC, confusion matrices, SHAP)
```

---

## 📊 Model Performance

Trained on **250,000 URLs** (PhishTank + OpenPhish + Tranco), evaluated on **37,500 held-out test URLs**:

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9999 | 0.9994 | 0.9997 | 1.0000 |
| **Ensemble** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

Ensemble on test set: **1 missed phishing URL out of 22,500** (FNR = 0.004%)

---

## 🔌 API Usage

### Authentication
```bash
# Use the demo API key directly as a Bearer token:
Authorization: Bearer demo-key-phishguard-2024

# Or exchange for a JWT:
POST /auth/token  {"api_key": "demo-key-phishguard-2024"}
```

### Single URL Analysis
```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer demo-key-phishguard-2024" \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-secure-verify.xyz/login.php", "include_explanation": true}'
```

**Response:**
```json
{
  "url": "http://paypal-secure-verify.xyz/login.php",
  "phishing_probability": 0.999,
  "risk_level": "High",
  "explanation": [
    {"feature": "tld_risk_score", "value": 0.95, "shap_value": 0.086,
     "impact": "increases_risk", "human_label": "High-risk top-level domain (.xyz)"},
    ...
  ],
  "text_summary": "🚨 Risk Level: High (99.9% phishing probability)\n...",
  "processing_time_ms": 117.4
}
```

### Batch Analysis (up to 100 URLs)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Authorization: Bearer demo-key-phishguard-2024" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://google.com", "http://phish.xyz/login"]}'
```

---

## 🔧 Training Your Own Models

```powershell
# Run data pipeline first (generates ~250k samples)
python -m data.pipeline --use-seed

# Train RF + XGBoost + Ensemble (fast, ~2 min)
python -m models.trainer --no-nn --sample 50000

# With neural network (requires: pip install torch)
python -m models.trainer --sample 50000

# With Optuna hyperparameter tuning for XGBoost
python -m models.trainer --tune-xgb --n-trials 50
```

---

## 🧩 Feature Groups (102 total)

| Group | Count | Examples |
|-------|-------|---------|
| URL Lexical | 45 | entropy, length, TLD risk, homographs, brand detection |
| Domain | 19 | WHOIS age, DNS A/MX/NS, SSL validity/issuer |
| Webpage | 31 | form actions, JS obfuscation, hidden iframes, meta-refresh |
| NLP | 7 | phishing phrase density, urgency words, brand impersonation |

---

## 🔑 API Keys (Demo)

| Key | Role |
|-----|------|
| `demo-key-phishguard-2024` | Read-only predictions |
| `admin-key-phishguard-9999` | Admin access |

> **Production**: Replace with database-backed API key management.

---

## 📦 Tech Stack

- **ML**: scikit-learn, XGBoost, PyTorch (optional)
- **Explainability**: SHAP (TreeExplainer + KernelExplainer)
- **API**: FastAPI + Uvicorn + Pydantic v2
- **Caching**: Redis (async, graceful degradation)
- **Auth**: JWT (python-jose) + API key
- **Frontend**: Vanilla HTML/CSS/JS (no framework, no build step)
- **Logging**: Loguru structured logs
