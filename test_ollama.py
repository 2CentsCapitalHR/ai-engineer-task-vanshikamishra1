import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.1:8b",
        "prompt": "Say 'Ollama is working!'",
        "stream": False
    }
    response = requests.post(url, json=data)
    print(response.json()["response"])

if __name__ == "__main__":
    test_ollama()