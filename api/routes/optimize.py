import threading
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.optimization_service import optimize_model


router = APIRouter()


class OptimizationRequest(BaseModel):
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
        description="Pruning fractions to evaluate.",
    )

    max_accuracy_loss: float = Field(
        default=0.50,
        ge=0.0,
        description=(
            "Maximum allowed accuracy loss "
            "in percentage points."
        ),
    )


class OptimizationJob:
    def __init__(self):
        self.status = "queued"
        self.stage = "queued"
        self.progress = 0
        self.message = "Optimization queued."

        self.candidate = None
        self.total_candidates = None

        self.epoch = None
        self.total_epochs = None

        self.accuracy = None
        self.loss = None

        self.result = None
        self.error = None


JOBS: dict[str, OptimizationJob] = {}

JOBS_LOCK = threading.Lock()


def _update_job(
    job_id,
    **updates,
):
    with JOBS_LOCK:
        job = JOBS[job_id]

        for key, value in updates.items():
            setattr(
                job,
                key,
                value,
            )


def _run_optimization(
    job_id,
    pruning_levels,
    max_accuracy_loss,
):
    _update_job(
        job_id,
        status="running",
        stage="starting",
        progress=0,
        message="Starting optimization.",
    )

    def progress_callback(
        stage,
        **data,
    ):
        total_candidates = data.get(
            "total_candidates"
        )

        candidate = data.get(
            "candidate"
        )

        progress = None

        if stage == "pruning":
            if (
                candidate is not None
                and total_candidates
            ):
                progress = (
                    ((candidate - 1)
                    / total_candidates)
                    * 100
                )

        elif stage == "fine_tuning":
            if (
                candidate is not None
                and total_candidates
            ):
                epoch = data.get(
                    "epoch",
                    0,
                )

                total_epochs = data.get(
                    "total_epochs",
                    1,
                )

                candidate_progress = (
                    (epoch / total_epochs)
                )

                progress = (
                    (
                        (candidate - 1)
                        + candidate_progress
                    )
                    / total_candidates
                ) * 100

        elif stage == "evaluating":
            if (
                candidate is not None
                and total_candidates
            ):
                progress = (
                    (
                        candidate
                        / total_candidates
                    )
                    * 100
                )

        elif stage == "selection":
            progress = 98

        elif stage == "completed":
            progress = 100

        updates = {
            "status": (
                "completed"
                if stage == "completed"
                else "running"
            ),
            "stage": stage,
            "message": data.get(
                "message",
                stage.replace(
                    "_",
                    " ",
                ).title(),
            ),
            "candidate": candidate,
            "total_candidates":
                total_candidates,
            "epoch": data.get(
                "epoch"
            ),
            "total_epochs":
                data.get(
                    "total_epochs"
                ),
            "accuracy":
                data.get(
                    "accuracy"
                ),
            "loss":
                data.get(
                    "loss"
                ),
        }

        if progress is not None:
            updates["progress"] = round(
                max(
                    0,
                    min(
                        100,
                        progress,
                    ),
                ),
                1,
            )

        if stage == "completed":
            updates["result"] = data.get(
                "best_model"
            )

        _update_job(
            job_id,
            **updates,
        )

    try:
        result = optimize_model(
            pruning_levels=pruning_levels,
            max_accuracy_loss=max_accuracy_loss,
            progress_callback=progress_callback,
        )

        _update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Optimization completed.",
            result=result,
        )

    except Exception as error:
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message="Optimization failed.",
            error=str(error),
        )


@router.post("/optimize")
def optimize(
    request: OptimizationRequest,
):
    for level in request.pruning_levels:
        if not 0.0 <= level < 1.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Each pruning level must be "
                    "between 0.0 and 1.0."
                ),
            )

    job_id = str(
        uuid.uuid4()
    )

    job = OptimizationJob()
    job.job_id = job_id
    with JOBS_LOCK:
        JOBS[job_id] = job

    worker = threading.Thread(
        target=_run_optimization,
        args=(
            job_id,
            request.pruning_levels,
            request.max_accuracy_loss,
        ),
        daemon=True,
    )

    worker.start()

    return {
        "job_id": job_id,
        "status": job.status,
        "message": job.message,
    }


@router.get(
    "/optimize/status/{job_id}"
)
def optimization_status(
    job_id: str,
):
    with JOBS_LOCK:
        job = JOBS.get(
            job_id
        )

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Optimization job not found.",
            )

        return {
            "job_id": job_id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "message": job.message,
            "candidate": job.candidate,
            "total_candidates":
                job.total_candidates,
            "epoch": job.epoch,
            "total_epochs":
                job.total_epochs,
            "accuracy": job.accuracy,
            "loss": job.loss,
            "result": job.result,
            "error": job.error,
        }