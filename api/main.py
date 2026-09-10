from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.model import (
    get_model_profile,
    router as model_router,
)
from api.routes.optimize import router as optimize_router
from api.routes.results import router as results_router


app = FastAPI(
    title="Neural Network Optimization Engine",
    description=(
        "API for profiling, pruning, and optimizing "
        "neural network models."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model/profile")
def model_profile():
    return get_model_profile()


app.include_router(model_router)
app.include_router(optimize_router)
app.include_router(results_router)