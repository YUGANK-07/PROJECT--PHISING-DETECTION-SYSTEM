import requests
import json

response = requests.post(
    'http://localhost:8000/predict',
    headers={'Authorization': 'Bearer demo-key-phishguard-2024'},
    json={'url': 'https://microsoft-login-secure.xyz', 'include_explanation': True}
)

with open('test_output.json', 'w', encoding='utf-8') as f:
    json.dump(response.json(), f, indent=2, ensure_ascii=False)
