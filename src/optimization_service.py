from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .finetune import fine_tune
from .model import MLP
from .optimizer import is_better_candidate, save_optimized_model
from .profiler import (
    count_nonzero_parameters,
    count_parameters,
    count_weight_parameters,
    measure_latency,
)
from .structured_pruning import structured_prune


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mlp.pth"
OPTIMIZED_MODEL_PATH = PROJECT_ROOT / "models" / "optimized_mlp.pth"
DATA_DIR = PROJECT_ROOT / "data"


BATCH_SIZE = 128
FINE_TUNE_EPOCHS = 2
FINE_TUNE_LEARNING_RATE = 0.0001


def evaluate(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


def get_model_size_mb(model):
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    return parameter_bytes / (1024 ** 2)


def calculate_sparsity(model):
    total = 0
    nonzero = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total += module.weight.numel()
            nonzero += torch.count_nonzero(
                module.weight
            ).item()

    if total == 0:
        return 0.0

    return 1.0 - (nonzero / total)


def load_baseline(device):
    model = MLP().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def optimize_model(
    pruning_levels,
    max_accuracy_loss=0.50,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    baseline_model = load_baseline(device)

    baseline_accuracy = evaluate(
        baseline_model,
        test_loader,
        device,
    )

    results = []

    best_candidate = None
    best_model = None

    for amount in pruning_levels:
        baseline_model = load_baseline(device)

        optimized_model = structured_prune(
            baseline_model,
            amount,
        )

        optimized_model.eval()

        before_accuracy = evaluate(
            optimized_model,
            test_loader,
            device,
        )

        fine_tune(
            optimized_model,
            train_loader,
            device,
            epochs=FINE_TUNE_EPOCHS,
            learning_rate=FINE_TUNE_LEARNING_RATE,
        )

        optimized_accuracy = evaluate(
            optimized_model,
            test_loader,
            device,
        )

        parameters = count_parameters(
            optimized_model
        )

        weight_parameters = count_weight_parameters(
            optimized_model
        )

        nonzero_parameters = count_nonzero_parameters(
            optimized_model
        )

        sparsity = calculate_sparsity(
            optimized_model
        )

        model_size = get_model_size_mb(
            optimized_model
        )

        latency = measure_latency(
            optimized_model,
            device,
        )

        accuracy_change = (
            optimized_accuracy
            - baseline_accuracy
        )

        result = {
            "pruning": amount,
            "hidden1_size": optimized_model.network[0].out_features,
            "hidden2_size": optimized_model.network[2].out_features,
            "parameters": parameters,
            "weight_parameters": weight_parameters,
            "nonzero_parameters": nonzero_parameters,
            "sparsity": sparsity,
            "model_size_mb": model_size,
            "accuracy": optimized_accuracy,
            "accuracy_change": accuracy_change,
            "latency_ms": latency,
            "before_finetuning_accuracy": before_accuracy,
        }

        results.append(result)

        if is_better_candidate(
            result,
            best_candidate,
            max_accuracy_loss=max_accuracy_loss,
        ):
            best_candidate = result
            best_model = optimized_model

    if best_candidate is None:
        raise ValueError(
            "No model satisfies the accuracy constraint."
        )

    save_optimized_model(
        best_model,
        OPTIMIZED_MODEL_PATH,
    )

    return {
        "device": str(device),
        "baseline_accuracy": baseline_accuracy,
        "max_accuracy_loss": max_accuracy_loss,
        "best_model": best_candidate,
        "results": results,
        "optimized_model_path": str(
            OPTIMIZED_MODEL_PATH
        ),
    }