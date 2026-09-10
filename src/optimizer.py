# src/optimizer.py

from pathlib import Path

import torch


def is_better_candidate(
    candidate,
    current_best,
    max_accuracy_loss=0.5,
):
    """
    Determine whether a candidate model should replace
    the current best model.

    A candidate is valid when its accuracy loss is within
    the configured limit.

    Among valid candidates, the model with fewer total
    trainable parameters is preferred.

    If parameter counts are equal, higher accuracy wins.
    """

    candidate_change = float(
        candidate["accuracy_change"]
    )

    candidate_loss = max(
        0.0,
        -candidate_change,
    )

    if candidate_loss > max_accuracy_loss:
        return False

    if current_best is None:
        return True

    current_change = float(
        current_best["accuracy_change"]
    )

    current_loss = max(
        0.0,
        -current_change,
    )

    if current_loss > max_accuracy_loss:
        return True

    candidate_parameters = int(
        candidate["parameters"]
    )

    current_parameters = int(
        current_best["parameters"]
    )

    if candidate_parameters < current_parameters:
        return True

    if candidate_parameters == current_parameters:
        return (
            float(candidate["accuracy"])
            > float(current_best["accuracy"])
        )

    return False


def save_optimized_model(
    model,
    model_path,
):
    """
    Save the optimized model state dictionary and
    architecture metadata.
    """

    model_path = Path(model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hidden1_size = model.network[0].out_features
    hidden2_size = model.network[2].out_features

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden1_size": hidden1_size,
        "hidden2_size": hidden2_size,
    }

    torch.save(
        checkpoint,
        model_path,
    )