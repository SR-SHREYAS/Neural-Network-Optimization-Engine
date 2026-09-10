import time
from pathlib import Path

import torch

from .model import MLP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mlp.pth"


def count_parameters(model):
    """Count all trainable parameters, including weights and biases."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_weight_parameters(model):
    """Count weight parameters in Linear layers."""
    total_parameters = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total_parameters += module.weight.numel()

    return total_parameters


def count_nonzero_parameters(model):
    """Count non-zero weights in Linear layers."""
    total_nonzero = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total_nonzero += torch.count_nonzero(
                module.weight
            ).item()

    return total_nonzero


def get_model_size_mb(model_path):
    size_bytes = Path(model_path).stat().st_size
    return size_bytes / (1024 ** 2)


def measure_latency(model, device, iterations=100):
    input_tensor = torch.randn(
        1,
        784,
        device=device,
    )

    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MLP().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    total_parameters = count_parameters(model)
    weight_parameters = count_weight_parameters(model)
    nonzero_parameters = count_nonzero_parameters(model)
    model_size = get_model_size_mb(MODEL_PATH)
    latency = measure_latency(model, device)

    print("Model: MLP")
    print(f"Total trainable parameters: {total_parameters:,}")
    print(f"Weight parameters: {weight_parameters:,}")
    print(f"Non-zero weights: {nonzero_parameters:,}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Inference latency: {latency:.3f} ms")


if __name__ == "__main__":
    main()