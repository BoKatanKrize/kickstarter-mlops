import json
import requests

with open("sample_kickstarter_project.json", "r", encoding="utf-8") as infile:
    sample = json.load(infile)

URL = "http://localhost:9696/predict"
# Convert the Python dictionary to a JSON string
sample_json = json.dumps(sample)
response = requests.post(URL, json=sample_json, timeout=10)
print(response.json())