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
from pruning import prune_model
from finetune import fine_tune
from profiler import (
    count_parameters,
    count_nonzero_parameters,
    measure_latency,
)


MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mlp.pth"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "pruning_results.csv"

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


def evaluate(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def calculate_sparsity(model):
    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.detach()

            total_weights += weights.numel()
            zero_weights += (weights == 0).sum().item()

    return zero_weights / total_weights


def load_model(device):
    model = MLP().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    return model


def save_results(results):
    RESULTS_DIR.mkdir(exist_ok=True)

    fieldnames = [
        "pruning",
        "sparsity",
        "accuracy",
        "accuracy_change",
        "total_parameters",
        "nonzero_parameters",
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


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

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

    baseline_model = load_model(device)

    baseline_accuracy = evaluate(
        baseline_model,
        test_loader,
        device,
    )

    baseline_parameters = count_parameters(
        baseline_model
    )

    baseline_nonzero_parameters = (
        count_nonzero_parameters(baseline_model)
    )

    baseline_sparsity = 1 - (
        baseline_nonzero_parameters
        / baseline_parameters
    )

    baseline_latency = measure_latency(
        baseline_model,
        device,
    )

    print()
    print("=== Baseline ===")
    print(f"Accuracy: {baseline_accuracy:.2f}%")
    print(f"Parameters: {baseline_parameters:,}")
    print(
        f"Non-zero parameters: "
        f"{baseline_nonzero_parameters:,}"
    )
    print(f"Sparsity: {baseline_sparsity * 100:.2f}%")
    print(f"Latency: {baseline_latency:.3f} ms")

    results = [
        {
            "pruning": 0.0,
            "sparsity": baseline_sparsity,
            "accuracy": baseline_accuracy,
            "accuracy_change": 0.0,
            "total_parameters": baseline_parameters,
            "nonzero_parameters": baseline_nonzero_parameters,
            "latency_ms": baseline_latency,
        }
    ]

    print()
    print("=== Starting pruning sweep ===")
    print()

    for pruning_amount in PRUNING_LEVELS:
        print(
            f"--- Pruning "
            f"{pruning_amount * 100:.0f}% ---"
        )

        model = load_model(device)

        prune_model(
            model,
            pruning_amount,
        )

        pruned_accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        print(
            f"Before fine-tuning: "
            f"{pruned_accuracy:.2f}%"
        )

        fine_tune(
            model,
            train_loader,
            device,
            epochs=FINE_TUNE_EPOCHS,
            learning_rate=FINE_TUNE_LEARNING_RATE,
        )

        fine_tuned_accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        final_sparsity = calculate_sparsity(
            model
        )

        total_parameters = count_parameters(
            model
        )

        nonzero_parameters = (
            count_nonzero_parameters(model)
        )

        latency = measure_latency(
            model,
            device,
        )

        accuracy_change = (
            fine_tuned_accuracy
            - baseline_accuracy
        )

        print(
            f"After fine-tuning: "
            f"{fine_tuned_accuracy:.2f}%"
        )

        print(
            f"Sparsity: "
            f"{final_sparsity * 100:.2f}%"
        )

        print(
            f"Non-zero parameters: "
            f"{nonzero_parameters:,}"
        )

        print(
            f"Latency: "
            f"{latency:.3f} ms"
        )

        print(
            f"Accuracy change: "
            f"{accuracy_change:+.2f} pp"
        )

        results.append(
            {
                "pruning": pruning_amount,
                "sparsity": final_sparsity,
                "accuracy": fine_tuned_accuracy,
                "accuracy_change": accuracy_change,
                "total_parameters": total_parameters,
                "nonzero_parameters": nonzero_parameters,
                "latency_ms": latency,
            }
        )

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.utils.prune.remove(
                    module,
                    "weight",
                )

        print()

    save_results(results)

    print("=== Pruning Sweep Results ===")
    print(
        f"{'Pruning':>8} "
        f"{'Sparsity':>9} "
        f"{'Accuracy':>10} "
        f"{'Change':>10} "
        f"{'Non-zero':>12} "
        f"{'Latency':>10}"
    )

    for result in results:
        print(
            f"{result['pruning'] * 100:>7.0f}% "
            f"{result['sparsity'] * 100:>8.2f}% "
            f"{result['accuracy']:>9.2f}% "
            f"{result['accuracy_change']:>+9.2f} pp "
            f"{result['nonzero_parameters']:>11,} "
            f"{result['latency_ms']:>9.3f} ms"
        )

    print()
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()