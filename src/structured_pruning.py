import torch

from .model import MLP


def _select_neurons(weight, amount):
    """
    Select the neurons with the largest outgoing L1 weight magnitude.

    Returns:
        indices of neurons to keep, sorted by importance.
    """
    num_neurons = weight.shape[0]
    keep_count = max(1, int(round(num_neurons * (1.0 - amount))))

    importance = weight.abs().sum(dim=1)
    _, indices = torch.topk(importance, keep_count)

    return indices.sort().values


def structured_prune(model, amount):
    """
    Create a smaller MLP by removing hidden neurons.

    The same pruning fraction is applied independently to both
    hidden layers.

    Example:
        784 -> 256 -> 128 -> 10
        amount=0.50
        becomes approximately:
        784 -> 128 -> 64 -> 10
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError("amount must be between 0.0 and 1.0")

    if amount == 0.0:
        return model

    first_layer = model.network[0]
    second_layer = model.network[2]
    third_layer = model.network[4]

    first_keep = _select_neurons(first_layer.weight, amount)
    second_keep = _select_neurons(second_layer.weight, amount)

    first_keep = first_keep.to(first_layer.weight.device)
    second_keep = second_keep.to(second_layer.weight.device)

    new_first_size = len(first_keep)
    new_second_size = len(second_keep)

    optimized_model = MLP(
        hidden1_size=new_first_size,
        hidden2_size=new_second_size,
    ).to(first_layer.weight.device)

    new_first_layer = optimized_model.network[0]
    new_second_layer = optimized_model.network[2]
    new_third_layer = optimized_model.network[4]

    with torch.no_grad():
        # First hidden layer:
        # keep selected output neurons.
        new_first_layer.weight.copy_(
            first_layer.weight[first_keep]
        )
        new_first_layer.bias.copy_(
            first_layer.bias[first_keep]
        )

        # Second hidden layer:
        # keep selected output neurons and only the surviving
        # inputs from the first hidden layer.
        new_second_layer.weight.copy_(
            second_layer.weight[second_keep][:, first_keep]
        )
        new_second_layer.bias.copy_(
            second_layer.bias[second_keep]
        )

        # Output layer:
        # keep only the surviving inputs from the second hidden layer.
        new_third_layer.weight.copy_(
            third_layer.weight[:, second_keep]
        )
        new_third_layer.bias.copy_(
            third_layer.bias
        )

    return optimized_model