import torch

from .model import MLP


def neuron_importance(weight):
    """
    Score each output neuron by the L1 magnitude of its incoming weights.

    ``weight`` has shape ``(out_features, in_features)``. Summing over
    ``dim=1`` adds the absolute values of the weights that arrive at each
    output neuron from the previous layer.

    This is a heuristic, not a causal measure of a neuron's contribution.
    """
    return weight.abs().sum(dim=1)


def select_neurons(weight, amount):
    """
    Select the neurons with the largest incoming L1 weight magnitude.

    Returns:
        keep_indices: sorted indices of the neurons to retain.
        importance: the importance score for every original neuron.
        keep_count: the number of neurons retained.
    """
    num_neurons = weight.shape[0]
    keep_count = max(1, int(round(num_neurons * (1.0 - amount))))

    importance = neuron_importance(weight)
    _, indices = torch.topk(importance, keep_count)

    return indices.sort().values, importance, keep_count


def describe_layer_pruning(weight, amount, bias=None, top_weights=6):
    """
    Build a JSON-friendly description of the neuron selection for one layer.

    The returned values come directly from the checkpoint weight matrix:
    importance scores, retained/removed indices, and a small worked example
    of the incoming weights for the cutoff neurons.
    """
    keep, importance, keep_count = select_neurons(weight, amount)
    keep_list = [int(index) for index in keep.detach().cpu().tolist()]
    keep_set = set(keep_list)
    removed_list = [
        index for index in range(int(weight.shape[0])) if index not in keep_set
    ]

    importance_list = [float(value) for value in importance.detach().cpu().tolist()]
    bias_values = (
        [float(value) for value in bias.detach().cpu().tolist()]
        if bias is not None
        else None
    )

    def _neuron_example(index):
        row = weight[index].detach().cpu()
        source_count = int(row.numel())
        sample_count = min(top_weights, source_count)
        top = torch.topk(row.abs(), sample_count)
        source_indices = [int(value) for value in top.indices.tolist()]
        return {
            "index": int(index),
            "importance": float(importance_list[index]),
            "bias": (
                float(bias_values[index])
                if bias_values is not None
                else None
            ),
            "top_weights": [
                {
                    "source_index": source_index,
                    "weight": float(row[source_index].item()),
                }
                for source_index in source_indices
            ],
        }

    lowest_retained = None
    if keep_list:
        lowest_index = min(keep_list, key=lambda index: importance_list[index])
        lowest_retained = _neuron_example(lowest_index)

    highest_removed = None
    if removed_list:
        highest_index = max(removed_list, key=lambda index: importance_list[index])
        highest_removed = _neuron_example(highest_index)

    return {
        "original_size": int(weight.shape[0]),
        "keep_count": int(keep_count),
        "removed_count": len(removed_list),
        "kept_indices": keep_list,
        "removed_indices": removed_list,
        "importance": importance_list,
        "lowest_retained": lowest_retained,
        "highest_removed": highest_removed,
    }


def _select_neurons(weight, amount):
    """Backward-compatible wrapper returning only the retained indices."""
    keep, _, _ = select_neurons(weight, amount)
    return keep


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