"""FastAPI service that serves predictions and exposes Prometheus metrics."""
import time

import joblib
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from train import MODEL_PATH, train

if not MODEL_PATH.exists():
    train()
bundle = joblib.load(MODEL_PATH)

REQUESTS = Counter("iris_requests_total", "Prediction requests received")
PREDICTIONS = Counter("iris_predictions_total", "Predictions by class", ["species"])
LATENCY = Histogram(
    "iris_prediction_latency_seconds",
    "Time taken to serve a prediction",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)

app = FastAPI(title="Iris Classifier API", version="1.0")


class Flower(BaseModel):
    sepal_length: float = Field(gt=0, examples=[5.1])
    sepal_width: float = Field(gt=0, examples=[3.5])
    petal_length: float = Field(gt=0, examples=[1.4])
    petal_width: float = Field(gt=0, examples=[0.2])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(flower: Flower):
    REQUESTS.inc()
    start = time.perf_counter()
    features = [[flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]]
    probabilities = bundle["model"].predict_proba(features)[0]
    best = int(probabilities.argmax())
    species = bundle["classes"][best]
    PREDICTIONS.labels(species=species).inc()
    LATENCY.observe(time.perf_counter() - start)
    return {"species": species, "confidence": round(float(probabilities[best]), 4)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
