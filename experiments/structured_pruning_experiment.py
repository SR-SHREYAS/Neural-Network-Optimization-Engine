# experiments/structured_pruning_experiment.py

import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from model import MLP
from structured_pruning import structured_prune
from finetune import fine_tune
from optimizer import (
    is_better_candidate,
    save_optimized_model,
)
from profiler import (
    count_parameters,
    count_weight_parameters,
    count_nonzero_parameters,
    measure_latency,
)


MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mlp.pth"
OPTIMIZED_MODEL_PATH = (
    PROJECT_ROOT / "models" / "optimized_mlp.pth"
)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_PATH = (
    RESULTS_DIR / "structured_pruning_results.csv"
)


BATCH_SIZE = 128

PRUNING_LEVELS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

FINE_TUNE_EPOCHS = 2
FINE_TUNE_LEARNING_RATE = 0.0001

MAX_ACCURACY_LOSS = 0.50


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
    """
    Estimate dense parameter storage based on the model's
    current architecture and tensor dtypes.

    This is not the serialized checkpoint file size.
    """

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    return parameter_bytes / (1024 ** 2)


def calculate_sparsity(model):
    """
    Calculate weight sparsity across Linear layers.
    """

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


def save_results(results):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        writer.writerows(results)


def load_baseline(device):
    model = MLP().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
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

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline_model = load_baseline(device)

    baseline_accuracy = evaluate(
        baseline_model,
        test_loader,
        device,
    )

    baseline_parameters = count_parameters(
        baseline_model
    )

    baseline_weight_parameters = count_weight_parameters(
        baseline_model
    )

    baseline_nonzero = count_nonzero_parameters(
        baseline_model
    )

    baseline_latency = measure_latency(
        baseline_model,
        device,
    )

    baseline_size = get_model_size_mb(
        baseline_model
    )

    baseline_hidden1_size = (
        baseline_model.network[0].out_features
    )

    baseline_hidden2_size = (
        baseline_model.network[2].out_features
    )

    print("\n=== Baseline ===")

    print(
        f"Accuracy: "
        f"{baseline_accuracy:.2f}%"
    )

    print(
        f"Architecture: "
        f"784 -> {baseline_hidden1_size} -> "
        f"{baseline_hidden2_size} -> 10"
    )

    print(
        f"Total trainable parameters: "
        f"{baseline_parameters:,}"
    )

    print(
        f"Weight parameters: "
        f"{baseline_weight_parameters:,}"
    )

    print(
        f"Non-zero weights: "
        f"{baseline_nonzero:,}"
    )

    print(
        f"Estimated parameter storage: "
        f"{baseline_size:.4f} MB"
    )

    print(
        f"Latency: "
        f"{baseline_latency:.3f} ms"
    )

    results = []

    results.append(
        {
            "pruning": 0.0,
            "hidden1_size": baseline_hidden1_size,
            "hidden2_size": baseline_hidden2_size,
            "parameters": baseline_parameters,
            "weight_parameters": baseline_weight_parameters,
            "nonzero_parameters": baseline_nonzero,
            "sparsity": 0.0,
            "model_size_mb": baseline_size,
            "accuracy": baseline_accuracy,
            "accuracy_change": 0.0,
            "latency_ms": baseline_latency,
        }
    )

    # ---------------------------------------------------------
    # Optimization state
    # ---------------------------------------------------------

    best_candidate = None
    best_model = None

    # ---------------------------------------------------------
    # Structured pruning sweep
    # ---------------------------------------------------------

    print(
        "\n=== Starting structured pruning sweep ==="
    )

    for amount in PRUNING_LEVELS:

        print(
            f"\n--- Structured pruning "
            f"{amount * 100:.0f}% ---"
        )

        baseline_model = load_baseline(device)

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

        before_accuracy = evaluate(
            optimized_model,
            test_loader,
            device,
        )

        print(
            f"Architecture: "
            f"784 -> {hidden1_size} -> "
            f"{hidden2_size} -> 10"
        )

        print(
            f"Before fine-tuning: "
            f"{before_accuracy:.2f}%"
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

        print(
            f"After fine-tuning: "
            f"{optimized_accuracy:.2f}%"
        )

        print(
            f"Total trainable parameters: "
            f"{parameters:,}"
        )

        print(
            f"Weight parameters: "
            f"{weight_parameters:,}"
        )

        print(
            f"Estimated parameter storage: "
            f"{model_size:.4f} MB"
        )

        print(
            f"Latency: "
            f"{latency:.3f} ms"
        )

        print(
            f"Accuracy change: "
            f"{accuracy_change:+.2f} pp"
        )

        result = {
            "pruning": amount,
            "hidden1_size": hidden1_size,
            "hidden2_size": hidden2_size,
            "parameters": parameters,
            "weight_parameters": weight_parameters,
            "nonzero_parameters": nonzero_parameters,
            "sparsity": sparsity,
            "model_size_mb": model_size,
            "accuracy": optimized_accuracy,
            "accuracy_change": accuracy_change,
            "latency_ms": latency,
        }

        results.append(result)

        # -----------------------------------------------------
        # Candidate selection
        # -----------------------------------------------------

        if is_better_candidate(
            result,
            best_candidate,
            max_accuracy_loss=MAX_ACCURACY_LOSS,
        ):
            best_candidate = result
            best_model = optimized_model

            print(
                "Current best candidate updated."
            )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    save_results(results)

    # ---------------------------------------------------------
    # Display sweep results
    # ---------------------------------------------------------

    print(
        "\n=== Structured Pruning Results ==="
    )

    print(
        f"{'Pruning':>8} "
        f"{'Architecture':>20} "
        f"{'Params':>10} "
        f"{'Accuracy':>10} "
        f"{'Change':>10} "
        f"{'Size MB':>10} "
        f"{'Latency':>10}"
    )

    for result in results:

        architecture = (
            f"784-"
            f"{result['hidden1_size']}-"
            f"{result['hidden2_size']}-10"
        )

        print(
            f"{result['pruning'] * 100:7.0f}% "
            f"{architecture:>20} "
            f"{result['parameters']:10,} "
            f"{result['accuracy']:9.2f}% "
            f"{result['accuracy_change']:+9.2f} pp "
            f"{result['model_size_mb']:9.4f} "
            f"{result['latency_ms']:9.3f}"
        )

    print(
        f"\nResults saved to "
        f"{RESULTS_PATH}"
    )

    # ---------------------------------------------------------
    # Final optimization result
    # ---------------------------------------------------------

    print("\n=== Optimization ===")

    print(
        f"Maximum allowed accuracy loss: "
        f"{MAX_ACCURACY_LOSS:.2f} pp"
    )

    if best_candidate is None:
        print(
            "\nOptimization failed: "
            "No model satisfies the accuracy constraint."
        )

        return

    # ---------------------------------------------------------
    # Save optimized model
    # ---------------------------------------------------------

    save_optimized_model(
        best_model,
        OPTIMIZED_MODEL_PATH,
    )

    print("\nBest model:")

    print(
        f"Pruning: "
        f"{best_candidate['pruning'] * 100:.0f}%"
    )

    print(
        f"Architecture: "
        f"784 -> "
        f"{best_candidate['hidden1_size']} -> "
        f"{best_candidate['hidden2_size']} -> 10"
    )

    print(
        f"Total trainable parameters: "
        f"{best_candidate['parameters']:,}"
    )

    print(
        f"Weight parameters: "
        f"{best_candidate['weight_parameters']:,}"
    )

    print(
        f"Accuracy: "
        f"{best_candidate['accuracy']:.2f}%"
    )

    print(
        f"Accuracy change: "
        f"{best_candidate['accuracy_change']:+.2f} pp"
    )

    print(
        f"Estimated parameter storage: "
        f"{best_candidate['model_size_mb']:.4f} MB"
    )

    print(
        f"Latency: "
        f"{best_candidate['latency_ms']:.3f} ms"
    )

    print(
        f"Optimized model saved to: "
        f"{OPTIMIZED_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()