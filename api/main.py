from fastapi import FastAPI

from api.routes.model import get_model_profile

app = FastAPI(
    title="Neural Network Optimization Engine",
    description="API for profiling, pruning, and optimizing neural network models.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model/profile")
def model_profile():
    return get_model_profile()