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


MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mlp.pth"
DATA_DIR = PROJECT_ROOT / "data"

BATCH_SIZE = 128
PRUNING_AMOUNT = 0.30


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = MLP().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    baseline_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")

    prune_model(model, PRUNING_AMOUNT)

    pruned_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print(
        f"Pruning amount: "
        f"{PRUNING_AMOUNT * 100:.0f}%"
    )
    print(f"Pruned accuracy: {pruned_accuracy:.2f}%")

    fine_tune(
        model,
        train_loader,
        device,
        epochs=2,
        learning_rate=0.0001,
    )

    fine_tuned_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    sparsity = calculate_sparsity(model)

    print(
        f"Fine-tuned accuracy: "
        f"{fine_tuned_accuracy:.2f}%"
    )
    print(
        f"Actual sparsity: "
        f"{sparsity * 100:.2f}%"
    )
    print(
        f"Accuracy change: "
        f"{fine_tuned_accuracy - baseline_accuracy:+.2f} "
        f"percentage points"
    )

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.utils.prune.remove(
                module,
                "weight",
            )


if __name__ == "__main__":
    main()