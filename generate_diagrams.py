import os
import base64
import zlib
import urllib.request
import urllib.error

# Dictionary of all 10 required diagrams
diagrams = {
    "1_System_Architecture": """
graph TD
    User([End User / Client]) -->|Submits URL| UI[Web Interface SPA]
    UI -->|HTTP POST| API[FastAPI Gateway]
    API <--> Cache[(Redis Cache)]
    API --> Extractor[Feature Extraction Engine]
    Extractor --> Lex[Lexical]
    Extractor --> Dom[Domain/DNS]
    Extractor --> Web[Webpage/HTML]
    Extractor --> NLP[Semantic NLP]
    Lex & Dom & Web & NLP --> Vector[125-D Feature Vector]
    Vector --> Models[Ensemble ML Pipeline]
    Models --> SHAP[SHAP Explainer]
    SHAP --> API
    """,

    "2_Data_Flow_Diagram_DFD": """
flowchart LR
    A[User] -->|Raw URL| B(API Gateway)
    B -->|URL Hash| C{Redis}
    C -->|Hit| D[Return Cache]
    C -->|Miss| E(Extraction Engine)
    E --> F1[Lexical Analyzer]
    E --> F2[WHOIS/DNS Client]
    E --> F3[Web Scraper]
    E --> F4[NLP Tokenizer]
    F1 & F2 & F3 & F4 --> G[Vector Assembler]
    G -->|Array| H(ML Predictor)
    H -->|Probability| I(Explainer)
    I -->|JSON Report| B
    B --> A
    """,

    "3_Use_Case_Diagram": """
usecaseDiagram
    actor User
    actor Admin
    
    usecase "Submit URL for Scan" as UC1
    usecase "View Threat Report" as UC2
    usecase "Read SHAP Explanations" as UC3
    usecase "Generate API Token" as UC4
    usecase "View System Metrics" as UC5
    usecase "Update Whitelist" as UC6
    
    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    
    Admin --> UC1
    Admin --> UC5
    Admin --> UC6
    """,

    "4_Flowchart": """
flowchart TD
    Start([Start]) --> Input[/Receive URL/]
    Input --> Whitelist{In Whitelist?}
    Whitelist -->|Yes| Safe[/Return Safe Status/]
    Whitelist -->|No| Cache{In Redis Cache?}
    Cache -->|Yes| RetCache[/Return Cached Status/]
    Cache -->|No| Extract[Extract 125 Features]
    Extract --> RF[Run Random Forest]
    Extract --> XGB[Run XGBoost]
    RF & XGB --> Meta[Run Logistic Meta-Learner]
    Meta --> SHAP[Generate Explanations]
    SHAP --> SaveCache[Save to Cache]
    SaveCache --> Out[/Return JSON Output/]
    Out --> Stop([Stop])
    Safe --> Stop
    RetCache --> Stop
    """,

    "5_Workflow_Pipeline": """
graph LR
    subgraph Data Collection
        Raw[Raw URLs] --> Clean[Sanitized Strings]
    end
    subgraph Feature Extraction
        Clean --> DNS[Network Queries]
        Clean --> Regex[Pattern Matching]
        Clean --> DOM[HTML Parsing]
    end
    subgraph Classification
        DNS & Regex & DOM --> Vec[Feature Array]
        Vec --> Ens[Ensemble Predictor]
    end
    subgraph Output Generation
        Ens --> Score[Probability Score]
        Ens --> Reason[SHAP Values]
        Score & Reason --> JSON[Final Payload]
    end
    """,

    "6_Database_Design": """
erDiagram
    API_KEYS {
        string key_id PK
        string username
        string hashed_key
        datetime created_at
        boolean is_active
    }
    LOGS {
        string log_id PK
        string url_scanned
        float phishing_score
        string risk_level
        datetime scanned_at
        string api_key_used FK
    }
    WHITELIST {
        string domain PK
        string added_by
        datetime date_added
    }
    API_KEYS ||--o{ LOGS : generates
    WHITELIST ||--o{ LOGS : bypasses
    """,

    "7_Model_Training_Pipeline": """
graph TD
    Raw[(250k URL Dataset)] --> Pre[Deduplication & Balancing]
    Pre --> Split[80/20 Stratified Split]
    Split --> Train[(Training Data)]
    Split --> Test[(Test Data)]
    
    Train --> Tune{Optuna Hyperparameter Tuning}
    Tune --> Best[Optimal XGB Params]
    
    Train --> RF[Train Random Forest]
    Best --> XGB[Train XGBoost]
    
    RF & XGB --> OOF[Out-Of-Fold Predictions]
    OOF --> Meta[Train Logistic Meta-Classifier]
    
    Meta --> Eval[Evaluate on Test Data]
    Test --> Eval
    Eval --> Metric[Precision/Recall/F1/AUC]
    Metric --> Artifacts[Save .joblib Models]
    """,

    "8_Process_Flow": """
sequenceDiagram
    participant C as Client UI
    participant A as FastAPI
    participant R as Redis Cache
    participant E as Extractor
    participant M as Ensemble Model
    participant S as SHAP
    
    C->>A: POST /predict {"url": "..."}
    A->>R: Check Cache
    R-->>A: Cache Miss
    A->>E: Start Extraction
    E-->>A: 125-D Vector
    A->>M: Predict(Vector)
    M-->>A: 99.8% Probability
    A->>S: Explain(Vector)
    S-->>A: Feature Contributions
    A->>R: Set Cache
    A-->>C: 200 OK (JSON Report)
    """,

    "9_Module_Interaction": """
graph TD
    Router[FastAPI Routers] --> Controller[Prediction Controller]
    Controller --> Auth[Security / JWT Module]
    Controller --> Caching[Redis Module]
    Controller --> Pipeline[Feature Pipeline Module]
    Pipeline --> Models[Joblib Model Artifacts]
    Pipeline --> Utils[Helpers / DNS Resolvers]
    Models --> Explainer[SHAP Module]
    """,

    "10_User_Interaction_Flow": """
stateDiagram-v2
    [*] --> Dashboard : User visits site
    Dashboard --> Loading : Submits URL
    Loading --> Error : Invalid URL / Timeout
    Loading --> Result : API Returns 200 OK
    Result --> SafeView : Score < 0.5
    Result --> PhishingView : Score > 0.5
    SafeView --> Dashboard : Scan Another
    PhishingView --> Dashboard : Scan Another
    PhishingView --> Details : Clicks "View Reasons"
    Details --> Dashboard
    """
}

def encode_kroki(text):
    # Fix 'usecaseDiagram' error - it's not standard mermaid, need to replace with something else or correct it.
    # Actually mermaid doesn't support 'usecaseDiagram' natively without issues in some Kroki versions. 
    # But wait, Mermaid has experimental requirement diagrams, but 'usecaseDiagram' is PlantUML. 
    # Let me fix the usecase diagram to be a flowchart mimicking a usecase.
    compressed = zlib.compress(text.encode('utf-8'))
    return base64.urlsafe_b64encode(compressed).decode('ascii')

def main():
    out_dir = "diagrams"
    os.makedirs(out_dir, exist_ok=True)
    
    # Fix the usecase diagram as Mermaid doesn't strictly have a usecase diagram type, but we can fake it with graph/flowchart
    diagrams["3_Use_Case_Diagram"] = """
flowchart LR
    User([End User])
    Admin([System Admin])
    
    UC1(Submit URL for Scan)
    UC2(View Threat Report)
    UC3(Read SHAP Explanations)
    UC4(Generate API Token)
    UC5(View System Metrics)
    UC6(Update Whitelist)
    
    User --- UC1
    User --- UC2
    User --- UC3
    User --- UC4
    
    Admin --- UC1
    Admin --- UC5
    Admin --- UC6
    """

    print("Generating diagrams via Kroki API...")
    for name, code in diagrams.items():
        print(f"Generating {name}.png...")
        try:
            payload = encode_kroki(code.strip())
            url = f"https://kroki.io/mermaid/png/{payload}"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                img_data = response.read()
                
            out_path = os.path.join(out_dir, f"{name}.png")
            with open(out_path, "wb") as f:
                f.write(img_data)
            print(f"  -> Saved {out_path}")
            
        except Exception as e:
            print(f"  -> Failed to generate {name}: {e}")

if __name__ == "__main__":
    main()
