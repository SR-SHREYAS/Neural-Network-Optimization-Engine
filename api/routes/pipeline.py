from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.pipeline_service import (
    prune,
    finetune,
    evaluate,
    select,
)


router = APIRouter(
    prefix="/pipeline",
)


class PruneRequest(BaseModel):
    pruning_levels: list[float] = Field(
        default=[
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
        ],
        min_length=1,
    )


class SelectRequest(BaseModel):
    max_accuracy_loss: float = Field(
        default=0.50,
        ge=0.0,
    )


@router.post("/prune")
def run_pruning(request: PruneRequest):
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
        return prune(request.pruning_levels)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.post("/finetune")
def run_finetuning():
    try:
        return finetune()

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.post("/evaluate")
def run_evaluation():
    try:
        return evaluate()

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.post("/select")
def run_selection(request: SelectRequest):
    try:
        return select(
            request.max_accuracy_loss
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error