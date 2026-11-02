# MLOps Experiment 8: Iris Classifier

End-to-end MLOps pipeline by **Kavish Purohit (E040, BTI DS Sem 11, E2 batch)**.

| Stage | Tool | Files |
|---|---|---|
| Version control | Git, GitHub | this repository |
| CI | GitHub Actions | `.github/workflows/ci.yml` |
| Container | Docker | `Dockerfile` |
| Orchestration | Kubernetes | `k8s/app.yaml` |
| Monitoring | Prometheus, Grafana | `k8s/prometheus.yaml`, `k8s/grafana.yaml` |
| Deployment | Streamlit | `app.py` |

## Run locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Docker

```bash
docker build -t iris-api:1.0 .
docker run -p 8000:8000 iris-api:1.0
```

## Kubernetes and monitoring

```bash
kubectl apply -f k8s/app.yaml -f k8s/prometheus.yaml -f k8s/grafana.yaml
kubectl -n mlops port-forward svc/iris-api 8000:8000
kubectl -n mlops port-forward svc/prometheus 9090:9090
kubectl -n mlops port-forward svc/grafana 3000:3000
python scripts/generate_traffic.py
```
