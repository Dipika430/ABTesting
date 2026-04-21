# app.py
import pickle
import random
import time
from fastapi import FastAPI

app = FastAPI()

model_A = pickle.load(open("model_A.pkl", "rb"))
model_B = pickle.load(open("model_B.pkl", "rb"))

# Feature flag ratio
TRAFFIC_SPLIT = 0.7  # 70% A, 30% B

metrics = {
    "A": {"count": 0, "time": 0},
    "B": {"count": 0, "time": 0}
}

@app.get("/predict")
def predict():
    data = [[5.1, 3.5, 1.4, 0.2]]  # sample input
    
    start = time.time()
    
    if random.random() < TRAFFIC_SPLIT:
        model = model_A
        version = "A"
    else:
        model = model_B
        version = "B"
    
    prediction = model.predict(data)
    
    latency = time.time() - start
    
    # Store metrics
    metrics[version]["count"] += 1
    metrics[version]["time"] += latency
    
    return {
        "model_version": version,
        "prediction": int(prediction[0]),
        "latency": latency
    }

@app.get("/metrics")
def get_metrics():
    return metrics