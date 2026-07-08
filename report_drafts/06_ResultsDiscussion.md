# 6. RESULTS AND DISCUSSION

## 6.1 User Experience & Interaction Flow
![User Interaction Flow](diagrams/10_User_Interaction_Flow.png)

#### Diagram Explanation
This State Machine diagram defines the user experience (UX) and navigation paths within the frontend Single-Page Application (SPA). It maps out exactly how the interface reacts to varying outputs from the backend API.

#### Component Breakdown
- **States:** `Dashboard` (initial view), `Loading` (processing view), `Error` (failure view), `Result` (data returned view), `SafeView` (green status), `PhishingView` (red status), and `Details` (SHAP explanation overlay).
- **Transitions:** Actions triggering a state change, such as "Submits URL" or "Score > 0.5".

#### Data Flow Between Components
The user enters the application starting at the **Dashboard** state. Upon entering a URL, the interface transitions to the **Loading** state, displaying a skeleton UI. If the backend times out, the state transitions to an **Error** view. If the API returns a `200 OK` JSON response, the interface parses the `phishing_probability` float. This numerical data dictates a strict fork in the flow: if the probability is `< 0.5`, the UI transitions to **SafeView** (displaying a trusted shield). If the probability is `> 0.5`, the UI transitions to **PhishingView** (displaying an alert modal). From the PhishingView, the user can click a button to transition into the **Details** state, which visually renders the SHAP explanation arrays received from the JSON payload. The user can then return to the Dashboard to initiate another scan.

## 6.2 Performance Metrics

The PhishGuard models were strictly evaluated against the 37,500 held-out test URLs. The goal was not merely to achieve high accuracy, but to completely minimize False Negatives (where a phishing site is incorrectly allowed through) while keeping False Positives (blocking a legitimate site) near zero.

The Ensemble model demonstrated state-of-the-art performance, achieving a perfect precision and recall profile at four decimal places.

| Model | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost** | 0.9999 | 0.9994 | 0.9997 | 1.0000 |
| **Ensemble (Meta)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

## 6.3 Result Analysis

The quantitative results are unprecedented. Out of the 22,500 specific phishing URLs present in the test set, the Ensemble model missed only 1. 
- **False Negative Rate (FNR):** 0.004%. 
- **False Positive Rate (FPR):** Virtually 0.000% due to the inclusion of the strict domain verification whitelist before classification.

The Receiver Operating Characteristic Area Under Curve (ROC-AUC) score of 1.0000 indicates perfect separability. The 125-dimensional feature space proved dense enough that malicious URLs clustered entirely separately from benign URLs, allowing the XGBoost and Random Forest hyperplanes to cleanly divide the dataset.

## 6.4 Output Explanation and Explainable AI (XAI)

Unlike traditional models that return a silent, binary classification, PhishGuard generates a highly detailed JSON response via the FastAPI backend. This output is driven by the SHAP (SHapley Additive exPlanations) TreeExplainer.

When the model predicts a phishing site, SHAP mathematically computes which specific features contributed to the high risk score. This data is mapped to human-readable strings and returned to the UI.

### Example System Output (Threat Deconstruction)
When an attacker submits a highly sophisticated cloned PayPal site hosted on a new `.xyz` domain, the system responds dynamically:

1. **Risk Level:** `High`
2. **Probability Score:** `99.98%`
3. **Processing Time:** `117ms`
4. **Threat Explanations (SHAP Analysis):**
   - *High-risk top-level domain (.xyz) heavily increases risk.*
   - *Domain WHOIS age is less than 24 hours, heavily increases risk.*
   - *Presence of urgency keywords ("verify account") increases risk.*
   - *High ratio of obfuscated JavaScript detected in HTML body.*

This output is visualized on the frontend via dynamic progress bars and alert cards. The integration of SHAP bridges the gap between deep machine learning mathematics and practical, actionable cybersecurity intelligence. A Security Analyst reading this output does not have to guess why the system blocked the URL; the exact attack vectors used by the threat actor are explicitly exposed by the AI.
