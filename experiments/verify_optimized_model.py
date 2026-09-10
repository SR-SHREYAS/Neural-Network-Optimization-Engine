import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from model import MLP


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "optimized_mlp.pth"
)

DATA_DIR = PROJECT_ROOT / "data"

BATCH_SIZE = 128


def evaluate(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Optimized model not found: "
            f"{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    hidden1_size = checkpoint[
        "hidden1_size"
    ]

    hidden2_size = checkpoint[
        "hidden2_size"
    ]

    print(
        "\n=== Optimized Model Checkpoint ==="
    )

    print(
        f"Architecture: "
        f"784 -> {hidden1_size} -> "
        f"{hidden2_size} -> 10"
    )

    model = MLP(
        hidden1_size=hidden1_size,
        hidden2_size=hidden2_size,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Total trainable parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Test accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        "\nOptimized model checkpoint "
        "loaded successfully."
    )


if __name__ == "__main__":
    main()