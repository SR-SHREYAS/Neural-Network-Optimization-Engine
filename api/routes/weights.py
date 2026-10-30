"""Read-only inspection endpoints for the educational frontend.

These routes expose a compact, truthful view of the checkpoints and the
MNIST experiment. They never return full model files and never invent
weights, indices, or measurements.
"""
from pathlib import Path
from typing import Literal, Optional

import torch
from fastapi import APIRouter, HTTPException, Query
from torchvision import datasets, transforms

from src.model import MLP
from src.optimization_service import analyze_model
from src.structured_pruning import describe_layer_pruning

router = APIRouter(tags=["model-inspection"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "models" / "baseline_mlp.pth"
OPTIMIZED_PATH = PROJECT_ROOT / "models" / "optimized_mlp.pth"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_COUNT = 5

_DATASET_CACHE = None


def _load_model(kind: Literal["baseline", "optimized"]):
    path = BASELINE_PATH if kind == "baseline" else OPTIMIZED_PATH
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{kind} model checkpoint is not available yet",
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            hidden1 = int(
                checkpoint.get(
                    "hidden1_size",
                    state["network.0.weight"].shape[0],
                )
            )
            hidden2 = int(
                checkpoint.get(
                    "hidden2_size",
                    state["network.2.weight"].shape[0],
                )
            )
        else:
            state = checkpoint
            hidden1 = int(state["network.0.weight"].shape[0])
            hidden2 = int(state["network.2.weight"].shape[0])

        model = MLP(hidden1_size=hidden1, hidden2_size=hidden2)
        model.load_state_dict(state)
        model.eval()
        return model
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to inspect {kind} checkpoint",
        ) from exc


def _linear_layers(model):
    return [module for module in model.network if isinstance(module, torch.nn.Linear)]


def _load_test_dataset():
    global _DATASET_CACHE

    if _DATASET_CACHE is None:
        try:
            _DATASET_CACHE = datasets.MNIST(
                root=DATA_DIR,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="MNIST test data is not available on this machine.",
            ) from exc

    return _DATASET_CACHE


def _forward(model, image):
    with torch.no_grad():
        logits = model(image.unsqueeze(0)).squeeze(0)
        probabilities = torch.softmax(logits, dim=0)

    prediction = int(logits.argmax().item())

    return {
        "logits": [round(float(value), 6) for value in logits.tolist()],
        "probabilities": [
            round(float(value), 6) for value in probabilities.tolist()
        ],
        "prediction": prediction,
        "confidence": round(float(probabilities[prediction].item()), 6),
    }


def _activation_stats(model, image):
    with torch.no_grad():
        first = torch.relu(model.network[0](image.view(1, -1)))
        second = torch.relu(model.network[2](first))

    def stats(values):
        return {
            "size": int(values.numel()),
            "mean": round(float(values.mean().item()), 6),
            "max": round(float(values.max().item()), 6),
            "fraction_zero": round(
                float((values == 0).float().mean().item()),
                6,
            ),
        }

    return {
        "hidden1": stats(first),
        "hidden2": stats(second),
    }


def _neuron_example(model, image, top_k=12):
    first_layer = model.network[0]
    pixels = image.view(-1)

    with torch.no_grad():
        pre_activation = first_layer(pixels)
        activation = torch.relu(pre_activation)
        neuron_index = int(activation.argmax().item())
        row = first_layer.weight[neuron_index].detach().cpu()
        contributions = row * pixels
        top = torch.topk(contributions.abs(), min(top_k, contributions.numel()))

        entries = []
        for source_index in top.indices.tolist():
            source_index = int(source_index)
            entries.append(
                {
                    "pixel_index": source_index,
                    "row": source_index // 28,
                    "col": source_index % 28,
                    "pixel_value": round(float(pixels[source_index].item()), 6),
                    "weight": round(float(row[source_index].item()), 6),
                    "contribution": round(
                        float(contributions[source_index].item()),
                        6,
                    ),
                }
            )

    return {
        "layer": "Hidden 1",
        "neuron_index": neuron_index,
        "layer_size": int(first_layer.out_features),
        "bias": round(float(first_layer.bias[neuron_index].item()), 6),
        "pre_activation": round(float(pre_activation[neuron_index].item()), 6),
        "activation": round(float(activation[neuron_index].item()), 6),
        "top_contributions": entries,
        "note": (
            "The neuron shown is the hidden-1 unit with the largest ReLU "
            "activation for this sample. The listed terms are the largest "
            "absolute contributions x_i * w_ji for that unit."
        ),
    }


@router.get("/model/weights")
def get_model_weights(
    model: Literal["baseline", "optimized"] = Query("baseline"),
):
    """Return architecture metadata, importance scores, and sampled weights."""
    loaded = _load_model(model)
    linear_layers = _linear_layers(loaded)
    architecture = [784, *(layer.out_features for layer in linear_layers)]
    layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]
    layers = []
    connections = []

    for index, size in enumerate(architecture):
        sampled_indices = list(range(min(SAMPLE_COUNT, int(size))))
        if index == 0:
            biases = []
            importance = None
        else:
            layer = linear_layers[index - 1]
            biases = [
                round(float(value), 6)
                for value in layer.bias.detach().cpu()[:SAMPLE_COUNT].tolist()
            ]
            importance = [
                round(float(value), 6)
                for value in layer.weight.detach().cpu().abs().sum(dim=1).tolist()
            ]

        layers.append(
            {
                "name": layer_names[index],
                "size": int(size),
                "sampled_indices": sampled_indices,
                "sampled_biases": biases,
                "importance": importance,
            }
        )

    for layer_index, layer in enumerate(linear_layers):
        weight = layer.weight.detach().cpu()
        source_count = min(SAMPLE_COUNT, weight.shape[1])
        target_count = min(SAMPLE_COUNT, weight.shape[0])
        sample = []
        for target_index in range(target_count):
            for source_index in range(source_count):
                sample.append(
                    {
                        "source_index": source_index,
                        "target_index": target_index,
                        "weight": round(
                            float(weight[target_index, source_index].item()),
                            6,
                        ),
                    }
                )
        connections.append(
            {
                "source_layer": layer_index,
                "target_layer": layer_index + 1,
                "weights": sample,
            }
        )

    return {
        "model": model,
        "architecture": [int(value) for value in architecture],
        "total_parameters": int(
            sum(parameter.numel() for parameter in loaded.parameters())
        ),
        "layers": layers,
        "connections": connections,
        "importance_method": (
            "Layer importance arrays are actual checkpoint values: "
            "I_j = sum_i |W_ji|, computed as weight.abs().sum(dim=1)."
        ),
        "sample_note": (
            "Each displayed connection is an actual checkpoint weight for the "
            "sampled first five units/features. The diagram is a readable "
            "sample, not a rendering of every connection."
        ),
    }


@router.get("/model/pruning")
def get_pruning_indices(
    level: float = Query(0.6, ge=0.0, lt=1.0),
):
    """Return exact retained/removed neuron indices for one pruning level."""
    baseline = _load_model("baseline")
    first_layer = baseline.network[0]
    second_layer = baseline.network[2]

    first_info = describe_layer_pruning(
        first_layer.weight,
        level,
        bias=first_layer.bias,
    )
    second_info = describe_layer_pruning(
        second_layer.weight,
        level,
        bias=second_layer.bias,
    )

    hidden1 = int(first_info["keep_count"])
    hidden2 = int(second_info["keep_count"])
    baseline_parameters = int(
        sum(parameter.numel() for parameter in baseline.parameters())
    )
    pruned_parameters = (
        784 * hidden1
        + hidden1
        + hidden1 * hidden2
        + hidden2
        + 10 * hidden2
        + 10
    )

    return {
        "level": float(level),
        "baseline_architecture": [784, 256, 128, 10],
        "pruned_architecture": [784, hidden1, hidden2, 10],
        "baseline_parameters": baseline_parameters,
        "pruned_parameters": int(pruned_parameters),
        "parameter_reduction_pct": round(
            (baseline_parameters - pruned_parameters)
            / baseline_parameters
            * 100,
            4,
        ),
        "layers": [
            {"name": "Hidden 1", **first_info},
            {"name": "Hidden 2", **second_info},
        ],
        "importance_method": (
            "I_j = sum_i |W_ji|, computed as weight.abs().sum(dim=1). "
            "This is the L1 magnitude of each neuron's incoming weights."
        ),
        "retention_rule": (
            "keep_count = max(1, round(n * (1 - level))); the top keep_count "
            "neurons by importance are retained and returned in original order."
        ),
        "note": (
            "The indices come from the current baseline checkpoint. The "
            "importance score is a heuristic and is not a causal measure of "
            "how much a neuron contributes to test accuracy."
        ),
    }


@router.get("/model/sample")
def get_sample_prediction(
    index: int = Query(0, ge=0, le=9999),
):
    """Return one real MNIST test sample and both models' predictions."""
    dataset = _load_test_dataset()
    image, label = dataset[index]
    baseline = _load_model("baseline")

    optimized = None
    if OPTIMIZED_PATH.exists():
        optimized = _load_model("optimized")

    pixels = (
        image.squeeze(0)
        .mul(255)
        .round()
        .to(torch.int)
        .tolist()
    )

    response = {
        "index": int(index),
        "label": int(label),
        "pixels": pixels,
        "baseline": {
            **_forward(baseline, image),
            "activation_stats": _activation_stats(baseline, image),
        },
        "optimized": (
            {
                **_forward(optimized, image),
                "activation_stats": _activation_stats(optimized, image),
            }
            if optimized is not None
            else None
        ),
        "neuron_example": _neuron_example(baseline, image),
        "note": (
            "Inference uses argmax over raw logits. Probabilities are the "
            "softmax of those logits and are shown for interpretation only."
        ),
    }

    return response


@router.get("/model/analysis")
def get_model_analysis(
    level: Optional[float] = Query(None, ge=0.0, lt=1.0),
):
    """Return measured baseline metrics and one pruning level's analysis."""
    try:
        return analyze_model(level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to evaluate the baseline checkpoint. "
                "MNIST test data or PyTorch may be unavailable."
            ),
        ) from exc
