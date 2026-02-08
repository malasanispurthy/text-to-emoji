"""Small example client that POSTs to the API's /predict endpoint."""
import json
import requests


def send_prediction(text: str, url: str = "http://127.0.0.1:8000/predict") -> dict:
    payload = {"text": text}
    resp = requests.post(url, json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()


if __name__ == '__main__':
    examples = [
        "I am thrilled and so happy",
        "I'm sad and depressed",
        "This made me furious",
    ]
    for ex in examples:
        print(f"Sending: {ex}")
        print(json.dumps(send_prediction(ex), ensure_ascii=False))
