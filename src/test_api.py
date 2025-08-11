import requests

# Basic test for /predict endpoint
url = "http://localhost:8000/predict"
resp = requests.post(url, json={"symptoms": ["itching", "skin rash"]})
print("Status:", resp.status_code)
print(resp.json())

# Health and readiness
print("/health:", requests.get("http://localhost:8000/health").json())
print("/ready:", requests.get("http://localhost:8000/ready").json())
