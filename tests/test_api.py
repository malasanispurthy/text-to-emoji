from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api import app


client = TestClient(app)


def test_predict_endpoint():
    payload = {"text": "I am so happy and excited"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "label" in body and "emoji" in body
    assert body["label"] == "joy"
