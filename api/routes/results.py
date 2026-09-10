import csv
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "structured_pruning_results.csv"
)


class ExperimentResult(BaseModel):
    pruning: float
    hidden1_size: int
    hidden2_size: int
    parameters: int
    weight_parameters: int
    nonzero_parameters: int
    sparsity: float
    model_size_mb: float
    accuracy: float
    accuracy_change: float
    latency_ms: float


@router.get(
    "/results",
    response_model=list[ExperimentResult],
)
def get_results():
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No optimization results found.",
        )

    results = []

    with open(
        RESULTS_PATH,
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            results.append(
                ExperimentResult(
                    pruning=round(
                        float(row["pruning"]),
                        4,
                    ),
                    hidden1_size=int(
                        row["hidden1_size"]
                    ),
                    hidden2_size=int(
                        row["hidden2_size"]
                    ),
                    parameters=int(
                        row["parameters"]
                    ),
                    weight_parameters=int(
                        row["weight_parameters"]
                    ),
                    nonzero_parameters=int(
                        row["nonzero_parameters"]
                    ),
                    sparsity=round(
                        float(row["sparsity"]),
                        4,
                    ),
                    model_size_mb=round(
                        float(row["model_size_mb"]),
                        4,
                    ),
                    accuracy=round(
                        float(row["accuracy"]),
                        2,
                    ),
                    accuracy_change=round(
                        float(row["accuracy_change"]),
                        2,
                    ),
                    latency_ms=round(
                        float(row["latency_ms"]),
                        4,
                    ),
                )
            )

    return results