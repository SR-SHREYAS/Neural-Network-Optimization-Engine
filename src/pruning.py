import torch
import torch.nn.utils.prune as prune


def prune_model(model, amount):
    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    return model
