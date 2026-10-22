from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import MLP
from .structured_pruning import structured_prune
from .finetune import fine_tune
from .profiler import (
    count_parameters,
    count_weight_parameters,
    count_nonzero_parameters,
    measure_latency,
)
from .optimizer import (
    is_better_candidate,
    save_optimized_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline_mlp.pth"
)

OPTIMIZED_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "optimized_mlp.pth"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "structured_pruning_results.csv"
)


BATCH_SIZE = 128

FINE_TUNE_EPOCHS = 2
FINE_TUNE_LEARNING_RATE = 0.0001


def _load_baseline(device):
    model = MLP().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


def _load_data():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader


def _evaluate(
    model,
    data_loader,
    device,
    progress_callback=None,
    candidate_index=None,
    candidate_total=None,
):
    model.eval()

    correct = 0
    total = 0

    total_batches = len(data_loader)

    for batch_index, (images, labels) in enumerate(
        data_loader,
        start=1,
    ):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if progress_callback is not None:
            progress_callback(
                stage="evaluating",
                candidate=candidate_index,
                total_candidates=candidate_total,
                batch=batch_index,
                total_batches=total_batches,
            )

    return 100.0 * correct / total


def _get_model_size_mb(model):
    parameter_bytes = sum(
        parameter.numel()
        * parameter.element_size()
        for parameter in model.parameters()
    )

    return parameter_bytes / (
        1024 ** 2
    )


def _model_metrics(
    model,
    device,
):
    return {
        "hidden1_size":
            model.network[0].out_features,

        "hidden2_size":
            model.network[2].out_features,

        "parameters":
            count_parameters(model),

        "weight_parameters":
            count_weight_parameters(model),

        "nonzero_parameters":
            count_nonzero_parameters(model),

        "sparsity": 0.0,

        "model_size_mb":
            _get_model_size_mb(model),

        "latency_ms":
            measure_latency(
                model,
                device,
            ),
    }


def optimize_model(
    pruning_levels,
    max_accuracy_loss,
    progress_callback=None,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader, test_loader = (
        _load_data()
    )

    baseline_model = _load_baseline(
        device
    )

    if progress_callback is not None:
        progress_callback(
            stage="baseline",
            message="Evaluating baseline model.",
        )

    baseline_accuracy = _evaluate(
        baseline_model,
        test_loader,
        device,
        progress_callback=None,
    )

    total_candidates = len(
        pruning_levels
    )

    results = []

    best_candidate = None
    best_model = None

    # ---------------------------------------------------------
    # PRUNING + FINE-TUNING + EVALUATION
    # ---------------------------------------------------------

    for index, amount in enumerate(
        pruning_levels,
        start=1,
    ):
        if progress_callback is not None:
            progress_callback(
                stage="pruning",
                candidate=index,
                total_candidates=total_candidates,
                pruning=amount,
                message=(
                    f"Generating {amount * 100:.0f}% "
                    f"pruned candidate."
                ),
            )

        baseline_model = _load_baseline(
            device
        )

        optimized_model = structured_prune(
            baseline_model,
            amount,
        )

        optimized_model.eval()

        hidden1_size = (
            optimized_model
            .network[0]
            .out_features
        )

        hidden2_size = (
            optimized_model
            .network[2]
            .out_features
        )

        before_accuracy = _evaluate(
            optimized_model,
            test_loader,
            device,
            progress_callback=None,
        )

        if progress_callback is not None:
            progress_callback(
                stage="fine_tuning",
                candidate=index,
                total_candidates=total_candidates,
                epoch=0,
                total_epochs=FINE_TUNE_EPOCHS,
                before_accuracy=before_accuracy,
                message=(
                    f"Fine-tuning candidate "
                    f"{index} of "
                    f"{total_candidates}."
                ),
            )

        def on_fine_tune_progress(
            stage,
            candidate,
            total_candidates,
            epoch,
            total_epochs,
            accuracy,
            loss,
        ):
            if progress_callback is not None:
                progress_callback(
                    stage=stage,
                    candidate=candidate,
                    total_candidates=total_candidates,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    accuracy=accuracy,
                    loss=loss,
                    pruning=amount,
                    message=(
                        f"Fine-tuning candidate "
                        f"{candidate} of "
                        f"{total_candidates}."
                    ),
                )

        fine_tune(
            optimized_model,
            train_loader,
            device,
            epochs=FINE_TUNE_EPOCHS,
            learning_rate=FINE_TUNE_LEARNING_RATE,
            progress_callback=on_fine_tune_progress,
            candidate_index=index,
            candidate_total=total_candidates,
        )

        optimized_model.eval()

        if progress_callback is not None:
            progress_callback(
                stage="evaluating",
                candidate=index,
                total_candidates=total_candidates,
                message=(
                    f"Evaluating candidate "
                    f"{index} of "
                    f"{total_candidates}."
                ),
            )

        optimized_accuracy = _evaluate(
            optimized_model,
            test_loader,
            device,
            progress_callback=None,
        )

        metrics = _model_metrics(
            optimized_model,
            device,
        )

        accuracy_change = (
            optimized_accuracy
            - baseline_accuracy
        )

        result = {
            "pruning": amount,
            "hidden1_size": hidden1_size,
            "hidden2_size": hidden2_size,
            "parameters": metrics[
                "parameters"
            ],
            "weight_parameters": metrics[
                "weight_parameters"
            ],
            "nonzero_parameters": metrics[
                "nonzero_parameters"
            ],
            "sparsity": metrics[
                "sparsity"
            ],
            "model_size_mb": metrics[
                "model_size_mb"
            ],
            "accuracy": optimized_accuracy,
            "accuracy_change": accuracy_change,
            "latency_ms": metrics[
                "latency_ms"
            ],
            "before_finetuning_accuracy":
                before_accuracy,
        }

        results.append(
            result
        )

        if is_better_candidate(
            result,
            best_candidate,
            max_accuracy_loss=max_accuracy_loss,
        ):
            best_candidate = result
            best_model = optimized_model

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    import csv

    fieldnames = [
        "pruning",
        "hidden1_size",
        "hidden2_size",
        "parameters",
        "weight_parameters",
        "nonzero_parameters",
        "sparsity",
        "model_size_mb",
        "accuracy",
        "accuracy_change",
        "latency_ms",
    ]

    with open(
        RESULTS_PATH,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    key: result[key]
                    for key in fieldnames
                }
            )

    # ---------------------------------------------------------
    # SELECTION
    # ---------------------------------------------------------

    if progress_callback is not None:
        progress_callback(
            stage="selection",
            message=(
                "Selecting the smallest valid model."
            ),
        )

    if best_candidate is None:
        raise ValueError(
            "No candidate satisfies the "
            "maximum accuracy-loss constraint."
        )

    save_optimized_model(
        best_model,
        OPTIMIZED_MODEL_PATH,
    )

    if progress_callback is not None:
        progress_callback(
            stage="completed",
            progress=100,
            best_model=best_candidate,
            message=(
                f"{best_candidate['pruning'] * 100:.0f}% "
                "pruning selected."
            ),
        )

    return {
        "device": str(device),
        "baseline_accuracy":
            baseline_accuracy,
        "max_accuracy_loss":
            max_accuracy_loss,
        "best_model":
            best_candidate,
        "results":
            results,
        "optimized_model_path":
            str(OPTIMIZED_MODEL_PATH),
    }