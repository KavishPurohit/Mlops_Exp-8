"""Streamlit front end for the Iris classifier."""
import json

import joblib
import pandas as pd
import streamlit as st

from train import METRICS_PATH, MODEL_PATH, train

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())


bundle, metrics = load_bundle()

st.title("🌸 Iris Flower Classifier")
st.caption("MLOps Experiment 8 | Kavish Purohit | E040 | BTI DS Sem 11 | E2 batch")

left, right = st.columns(2)
sepal_length = left.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
sepal_width = left.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = right.slider("Petal length (cm)", 1.0, 7.0, 4.3)
petal_width = right.slider("Petal width (cm)", 0.1, 2.5, 1.3)

features = [[sepal_length, sepal_width, petal_length, petal_width]]
probabilities = bundle["model"].predict_proba(features)[0]
species = bundle["classes"][int(probabilities.argmax())]

st.success(f"Predicted species: **{species}** ({probabilities.max():.0%} confidence)")
st.bar_chart(pd.DataFrame({"Probability": probabilities}, index=bundle["classes"]))

accuracy_col, f1_col = st.columns(2)
accuracy_col.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
f1_col.metric("Macro F1", f"{metrics['f1_macro']:.2%}")
