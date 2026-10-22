from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import MLP
from .structured_pruning import structured_prune
from .finetune import fine_tune
from .profiler import (
    count_nonzero_parameters,
    count_parameters,
    count_weight_parameters,
    measure_latency,
)
from .optimizer import (
    is_better_candidate,
    save_optimized_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mlp.pth"
OPTIMIZED_MODEL_PATH = PROJECT_ROOT / "models" / "optimized_mlp.pth"
DATA_DIR = PROJECT_ROOT / "data"

BATCH_SIZE = 128
FINE_TUNE_EPOCHS = 2
FINE_TUNE_LEARNING_RATE = 0.0001


class PipelineState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.baseline_accuracy = None
        self.max_accuracy_loss = 0.50

        self.pruned_candidates = []
        self.finetuned_candidates = []
        self.evaluated_candidates = []

        self.best_model = None
        self.stage = "idle"


STATE = PipelineState()


def _load_baseline():
    model = MLP().to(STATE.device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=STATE.device,
    )

    model.load_state_dict(state_dict)
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


def _evaluate(model, data_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(STATE.device)
            labels = labels.to(STATE.device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


def _model_metrics(model):
    return {
        "hidden1_size": model.network[0].out_features,
        "hidden2_size": model.network[2].out_features,
        "parameters": count_parameters(model),
        "weight_parameters": count_weight_parameters(model),
        "nonzero_parameters": count_nonzero_parameters(model),
        "model_size_mb": (
            sum(
                parameter.numel()
                * parameter.element_size()
                for parameter in model.parameters()
            )
            / (1024 ** 2)
        ),
        "latency_ms": measure_latency(
            model,
            STATE.device,
        ),
    }


def prune(pruning_levels):
    STATE.reset()

    baseline_model = _load_baseline()
    _, test_loader = _load_data()

    STATE.baseline_accuracy = _evaluate(
        baseline_model,
        test_loader,
    )

    candidates = []

    for level in pruning_levels:
        candidate = structured_prune(
            baseline_model,
            level,
        )

        candidate.eval()

        accuracy = _evaluate(
            candidate,
            test_loader,
        )

        metrics = _model_metrics(candidate)

        candidates.append(
            {
                "model": candidate,
                "pruning": level,
                "before_finetuning_accuracy": accuracy,
                **metrics,
            }
        )

    STATE.pruned_candidates = candidates
    STATE.stage = "pruned"

    return {
        "stage": STATE.stage,
        "device": str(STATE.device),
        "baseline_accuracy": STATE.baseline_accuracy,
        "candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key != "model"
            }
            for candidate in candidates
        ],
    }


def finetune():
    if not STATE.pruned_candidates:
        raise ValueError(
            "Run pruning before fine-tuning."
        )

    train_loader, _ = _load_data()

    finetuned = []

    for candidate in STATE.pruned_candidates:
        model = candidate["model"]

        fine_tune(
            model,
            train_loader,
            STATE.device,
            epochs=FINE_TUNE_EPOCHS,
            learning_rate=FINE_TUNE_LEARNING_RATE,
        )

        model.eval()

        finetuned.append(
            {
                **candidate,
                "model": model,
            }
        )

    STATE.finetuned_candidates = finetuned
    STATE.stage = "finetuned"

    return {
        "stage": STATE.stage,
        "candidate_count": len(finetuned),
        "message": (
            "Fine-tuning completed for all pruned candidates."
        ),
    }


def evaluate():
    if not STATE.finetuned_candidates:
        raise ValueError(
            "Run pruning and fine-tuning before evaluation."
        )

    _, test_loader = _load_data()

    evaluated = []

    for candidate in STATE.finetuned_candidates:
        model = candidate["model"]

        accuracy = _evaluate(
            model,
            test_loader,
        )

        metrics = _model_metrics(model)

        accuracy_change = (
            accuracy - STATE.baseline_accuracy
        )

        evaluated.append(
            {
                **candidate,
                "accuracy": accuracy,
                "accuracy_change": accuracy_change,
                **metrics,
            }
        )

    STATE.evaluated_candidates = evaluated
    STATE.stage = "evaluated"

    return {
        "stage": STATE.stage,
        "baseline_accuracy": STATE.baseline_accuracy,
        "results": [
            {
                key: value
                for key, value in candidate.items()
                if key != "model"
            }
            for candidate in evaluated
        ],
    }


def select(max_accuracy_loss):
    if not STATE.evaluated_candidates:
        raise ValueError(
            "Run pruning, fine-tuning and evaluation before selection."
        )

    STATE.max_accuracy_loss = max_accuracy_loss

    best_candidate = None
    best_model = None

    for candidate in STATE.evaluated_candidates:
        if is_better_candidate(
            candidate,
            best_candidate,
            max_accuracy_loss=max_accuracy_loss,
        ):
            best_candidate = candidate
            best_model = candidate["model"]

    if best_candidate is None:
        raise ValueError(
            "No candidate satisfies the maximum accuracy-loss constraint."
        )

    save_optimized_model(
        best_model,
        OPTIMIZED_MODEL_PATH,
    )

    STATE.best_model = best_candidate
    STATE.stage = "selected"

    result = {
        key: value
        for key, value in best_candidate.items()
        if key != "model"
    }

    return {
        "stage": STATE.stage,
        "baseline_accuracy": STATE.baseline_accuracy,
        "max_accuracy_loss": max_accuracy_loss,
        "best_model": result,
    }