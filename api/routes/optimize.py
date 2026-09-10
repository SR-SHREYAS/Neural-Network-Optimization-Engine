from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.optimization_service import optimize_model


router = APIRouter()


class OptimizationRequest(BaseModel):
    pruning_levels: list[float] = Field(
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        min_length=1,
        description="Pruning fractions to evaluate.",
    )

    max_accuracy_loss: float = Field(
        default=0.50,
        ge=0.0,
        description="Maximum allowed accuracy loss in percentage points.",
    )


class OptimizationResult(BaseModel):
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
    before_finetuning_accuracy: float


class OptimizationResponse(BaseModel):
    device: str
    baseline_accuracy: float
    max_accuracy_loss: float
    best_model: OptimizationResult
    results: list[OptimizationResult]


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
)
def optimize(request: OptimizationRequest):
    for level in request.pruning_levels:
        if not 0.0 <= level < 1.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Each pruning level must be "
                    "between 0.0 and 1.0."
                ),
            )

    try:
        result = optimize_model(
            pruning_levels=request.pruning_levels,
            max_accuracy_loss=request.max_accuracy_loss,
        )

        result.pop("optimized_model_path", None)

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error