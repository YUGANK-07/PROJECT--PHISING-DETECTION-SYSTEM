import urllib.request
import json
import ssl

req = urllib.request.Request(
    "http://localhost:8000/preview",
    data=json.dumps({"url": "https://example.com"}).encode('utf-8'),
    headers={"Authorization": "Bearer demo-key-phishguard-2024", "Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req) as response:
        print("Status:", response.status)
        print("Response:", response.read()[:100])
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print("Error Body:", e.read())
except Exception as e:
    print("Exception:", e)
