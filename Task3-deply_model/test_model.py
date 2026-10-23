import requests
import json
import os
url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
  "model": "accounts/pyroworks/deployments/kor9fm3p",
  "max_tokens": 1024,
  "top_k": 40,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "messages": [
    {
      "role": "user",
      "content": "Do you know if mintlify MCP exists?"
    }
  ]
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": f"Bearer {os.getenv('FIREWORKS_API_KEY')}"
}
response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
print(response.json()['choices'][0]['message']['content'])