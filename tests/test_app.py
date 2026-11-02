import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from api import app
from train import train

client = TestClient(app)
SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def test_model_accuracy_above_threshold():
    assert train()["accuracy"] >= 0.85


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_setosa():
    response = client.post("/predict", json=SETOSA)
    assert response.status_code == 200
    assert response.json()["species"] == "setosa"


def test_predict_rejects_invalid_input():
    assert client.post("/predict", json={**SETOSA, "petal_width": -1}).status_code == 422


def test_metrics_exposed():
    client.post("/predict", json=SETOSA)
    assert "iris_predictions_total" in client.get("/metrics").text
