"""Send sample prediction requests so the Grafana dashboard has data to show."""
import json
import random
import sys
import time
import urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/predict"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 300

SAMPLES = [(5.0, 3.4, 1.5, 0.2), (5.9, 2.8, 4.3, 1.3), (6.6, 3.0, 5.6, 2.1)]
KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

for _ in range(count):
    values = [round(v + random.uniform(-0.3, 0.3), 2) for v in random.choice(SAMPLES)]
    body = json.dumps(dict(zip(KEYS, values))).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    urllib.request.urlopen(request).read()
    time.sleep(0.2)

print(f"Sent {count} requests to {url}")
