import asyncio
from fastapi.testclient import TestClient
from api.main import app
from api.routers.predict import set_model

# Run with lifespan
with TestClient(app) as client:
    # Need auth token for predict
    resp = client.post("/auth/token", json={"api_key": "demo-key-phishguard-2024"})
    token = resp.json()["access_token"]
    
    # Predict request with visual verification
    predict_resp = client.post(
        "/predict", 
        headers={"Authorization": f"Bearer {token}"},
        json={
            "url": "http://paypal-secure-verify.xyz/login.php",
            "reference_url": "https://www.paypal.com",
            "include_explanation": False
        }
    )
    print("PREDICT RESP:", predict_resp.json())
