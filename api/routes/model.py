from pathlib import Path

import torch

from src.model import MLP
from src.profiler import (
    count_nonzero_parameters,
    count_parameters,
    count_weight_parameters,
    get_model_size_mb,
    measure_latency,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mlp.pth"


def get_model_profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model_size_mb = get_model_size_mb(MODEL_PATH)
    latency_ms = measure_latency(model, device)

    return {
        "model": "MLP",
        "architecture": "784 -> 256 -> 128 -> 10",
        "device": str(device),
        "total_parameters": total_parameters,
        "weight_parameters": weight_parameters,
        "nonzero_weights": nonzero_parameters,
        "model_size_mb": round(model_size_mb, 4),
        "inference_latency_ms": round(latency_ms, 4),
    }