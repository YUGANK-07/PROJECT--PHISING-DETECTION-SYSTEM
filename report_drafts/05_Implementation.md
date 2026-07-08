# 5. IMPLEMENTATION

## 5.1 Tools & Technologies Used

The implementation of PhishGuard spans a highly diverse technology stack, deliberately selected for enterprise-grade performance, maintainability, and scalability.

| Component / Layer | Technologies & Frameworks Utilized |
| :--- | :--- |
| **Machine Learning Core** | Python 3.10+, Scikit-Learn, XGBoost, PyTorch |
| **Explainable AI (XAI)** | SHAP (SHapley Additive exPlanations) |
| **Backend Framework** | FastAPI, Uvicorn, Pydantic v2 |
| **Caching Layer** | Redis (Asynchronous configuration) |
| **Security & Auth** | JSON Web Tokens (python-jose), API Key Middleware |
| **Frontend UI** | Vanilla HTML5, CSS3 (Glassmorphism design), JavaScript |
| **Data Engineering** | Pandas, NumPy, BeautifulSoup4, python-whois, dnspython |
| **Visual Subsystem** | Playwright (Headless browsing), ImageHash |

## 5.2 Database & Caching Schema
![Database Design](diagrams/6_Database_Design.png)

#### Diagram Explanation
This Entity-Relationship Diagram (ERD) defines the fundamental logical structure of the persistent data storage used by the PhishGuard backend, establishing the relational links between authentication, logging, and bypass lists.

#### Component Breakdown
- **API_KEYS Table:** Stores valid user credentials (`key_id`, `hashed_key`, `is_active`) to authenticate REST requests.
- **LOGS Table:** An immutable ledger storing historical scan data (`url_scanned`, `phishing_score`, `scanned_at`).
- **WHITELIST Table:** A registry of strictly trusted domains (`domain`, `added_by`) that bypass ML inspection.

#### Data Flow Between Components
The ERD illustrates one-to-many relationships. A single active **API_KEYS** entity can generate multiple **LOGS** (tracking which user scanned which URL). Conversely, a single entry in the **WHITELIST** table can bypass inspection for multiple **LOGS**, significantly altering the operational data flow by ensuring known safe traffic is never routed to the computationally expensive ML models.

## 5.3 Dataset Description

A robust machine learning model is entirely dependent on the quality and volume of its training data. PhishGuard was trained on a massive, highly curated dataset containing **250,000 URLs**.

- **Phishing Samples:** Sourced in real-time from active intelligence feeds including PhishTank and OpenPhish. This ensured the model learned from current, active threats rather than outdated, historical phishing campaigns.
- **Benign Samples:** Sourced from the Tranco Top 1 Million list. This guarantees that the model recognizes the complex architectures of legitimate, high-traffic domains (e.g., Google, Microsoft, AWS) and does not falsely flag them.
- **Data Splitting:** The data underwent rigorous deduplication and balancing before being split using an 80/20 stratified shuffle. The held-out test set of **37,500 URLs** was completely quarantined during training to prevent data leakage and guarantee unbiased performance evaluation.

## 5.4 Feature Engineering

The `FeaturePipeline` is the most critical software component of PhishGuard. It generates a 125-dimensional vector encompassing the following categories:

### Figure 5.4: Feature Engineering Workflow
![Workflow Pipeline](diagrams/5_Workflow_Pipeline.png)

#### Diagram Explanation
This workflow diagram demonstrates the chronological progression of data processing required to convert a simple text string (a URL) into a complex mathematical array that a machine learning model can understand.

#### Component Breakdown
- **Data Collection:** The initial stage containing raw URL strings and sanitation logic.
- **Feature Extraction:** Parallel modules executing Network Queries (DNS), Pattern Matching (Regex), and HTML Parsing (DOM).
- **Classification:** The convergence point where extracted features form a numerical Feature Array, which is fed to the Ensemble Predictor.
- **Output Generation:** The final calculation of probability scores and SHAP reasonings.

#### Data Flow Between Components
The flow initiates with **Raw URLs** moving through a sanitation filter into **Sanitized Strings**. These strings are simultaneously broadcast to three distinct extraction pods: **Network Queries**, **Regex**, and **DOM Parsing**. The physical outputs of these pods—such as HTTP status codes, character entropy floats, and hidden iframe counts—flow down and converge into a singular, dense **Feature Array**. This array flows directly into the **Ensemble Predictor**. The predictor splits its execution, simultaneously emitting a **Probability Score** and **SHAP Values**. Finally, these metrics merge into the **Final Payload** (JSON object).

### 5.4.1 URL Lexical Features (45 Dimensions)
Analyzes the mathematical and structural properties of the URL string itself without making network requests.
- **Shannon Entropy:** Detects algorithmically generated domains (DGAs) by measuring the randomness of the string.
- **TLD Risk Scoring:** Assigns higher risk weights to Top-Level Domains frequently abused by attackers (e.g., `.xyz`, `.top`, `.tk`).
- **Homograph Detection:** Identifies internationalized domain names (IDN) that look visually identical to trusted brands but use Cyrillic or Greek characters (e.g., `аpple.com`).
- **Brand Subdomain Abuse:** Detects patterns where attackers place a trusted brand in the subdomain of a malicious root (e.g., `paypal.login.secure-update-123.com`).

### 5.4.2 Domain & DNS Features (19 Dimensions)
Queries global internet registries to establish trust.
- **WHOIS Registration Age:** Phishing domains are often hours or days old. Legitimate domains are usually years old.
- **DNS Records Validation:** Checks for the presence of valid A, MX (Mail Exchange), and NS (Name Server) records. A domain without MX records is highly suspicious.
- **SSL Certificate Validation:** Checks the issuer. While attackers now use free SSL (like Let's Encrypt), the absence of SSL or the use of self-signed certificates heavily penalizes the score.

### 5.4.3 Webpage Structural Features (31 Dimensions)
Downloads the HTML/JS payload and structurally analyzes it.
- **Hidden Iframes:** Attackers often render the actual phishing form inside an invisible 1x1 pixel iframe to bypass text scrapers.
- **Form Action Hijacking:** Checks if HTML `<form>` tags are submitting data to suspicious, cross-origin domains or PHP scripts on compromised servers.
- **JavaScript Obfuscation:** Detects unusually long lines of JavaScript, excessive use of `eval()`, or heavy Base64 encoding.

### 5.4.4 Natural Language Processing (NLP) Features (7+ Dimensions)
Reads the visible text on the page as a human would.
- **Urgency Semantic Detection:** Uses TF-IDF and NLP to detect high-stress phrases such as "Account suspended", "Verify immediately", or "Unauthorised login attempt".

### 5.4.5 Visual Similarity
- **Screenshot Hashing:** The system utilizes Playwright to render the page in a headless browser. It generates a perceptual hash (pHash) of the screenshot and compares it against known legitimate brand templates (e.g., the real Microsoft login page) using a PyTorch ResNet model.

## 5.5 Model Training & Workflow

The model training pipeline (`trainer.py`) is fully automated. Both Random Forest and XGBoost undergo iterative training. XGBoost utilizes **Optuna**, a hyperparameter optimization framework, running 50 specific trials to fine-tune the learning rate (`eta`), maximum tree depth, and L1/L2 regularization to prevent overfitting on the noise inherent in the WHOIS data.

### Figure 5.5: Model Training Pipeline Diagram

![Model Training Pipeline](diagrams/7_Model_Training_Pipeline.png)

#### Diagram Explanation
This diagram outlines the exhaustive, automated machine learning training cycle, from raw dataset ingestion through validation and serialization of the final binary model artifacts.

#### Component Breakdown
- **Raw Data Node:** The 250k URL database.
- **Pre-Processing & Splitting:** Algorithms for deduplication, balancing, and 80/20 stratification.
- **Hyperparameter Tuning:** The Optuna framework loop evaluating XGBoost parameter permutations.
- **Base Models:** The Random Forest and XGBoost training execution blocks.
- **Meta-Classifier:** The Logistic Regression algorithm generating the ensemble layer.
- **Evaluation:** Testing against holdout data to calculate F1, Precision, and Recall metrics.

#### Data Flow Between Components
The physical data flow begins at the **250k URL Dataset**, progressing through **Pre-Processing** and splitting into two isolated partitions: **Training Data** (80%) and **Test Data** (20%). The Training Data feeds into an **Optuna Tuning** loop, which iterates repeatedly to output the **Optimal XGB Params**. The Training Data is then fed simultaneously to the **Random Forest** and **XGBoost** training modules. Both models output **Out-Of-Fold Predictions** (a matrix of probabilities), which flows directly into the **Logistic Meta-Classifier** for stacking. Once the Meta-Classifier is trained, the completely unseen **Test Data** is passed into it. The predictions are compared against the truth labels to generate the **Metrics**, and finally, the validated models are serialized into **.joblib files**.
