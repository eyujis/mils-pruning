import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy
from mils_pruning.complexity import BDMComplexityCalc, EntropyComplexityCalc


class Pruner:
    """
    Abstract base class for model pruners.
    Subclasses must implement the `prune` method.
    """
    def prune(self, model: nn.Module, n_nodes: int) -> nn.Module:
        raise NotImplementedError


def get_layer_sequence(model):
    """
    Extracts an ordered list of (name, param) tuples corresponding to
    the linear weight matrices in the model.

    Assumes:
        - All prunable layers are 2D (i.e., nn.Linear)
        - Weights are identified by "weight" in their parameter name
        - Layers are registered in forward order (as is typical in nn.Module)

    Returns:
        List of (name, param) tuples.
    """
    return [
        (name, param)
        for name, param in model.named_parameters()
        if "weight" in name and param.requires_grad and param.ndim == 2
    ]


class RandomPruner(Pruner):
    """
    Randomly removes `n_nodes` active neurons across the model by zeroing:
    - the row of the current layer
    - the corresponding column in the next layer
    """
    def prune(self, model: nn.Module, n_nodes: int) -> nn.Module:
        model = deepcopy(model)
        layers = get_layer_sequence(model)

        pruned = 0
        while pruned < n_nodes:
            # Collect all remaining unpruned neurons across all layers
            candidates = []
            for layer_idx in range(len(layers) - 1):
                name, param = layers[layer_idx]
                num_rows = param.shape[0]

                for i in range(num_rows):
                    if not torch.all(param[i] == 0.0):  # only unpruned neurons
                        candidates.append((layer_idx, i))

            if not candidates:
                break  # no more neurons to prune

            # Randomly choose one to prune
            layer_idx, i = random.choice(candidates)

            curr_param = layers[layer_idx][1]
            next_param = layers[layer_idx + 1][1]

            curr_param.data[i] = 0.0
            next_param.data[:, i] = 0.0
            pruned += 1

        return model



class MILSPruner(Pruner):
    """
    Prunes neurons using MILS with configurable strategy:
    - 'min_increase': prune neuron that causes the smallest complexity increase (standard MILS).
    - 'neutral_or_decrease': prune neuron with lowest absolute delta (near-zero or negative).
    - 'max_decrease': prune neuron that maximally reduces complexity.
    """
    def __init__(self, method="bdm", strategy="min_increase"):
        if method == "bdm":
            self.calc_cls = BDMComplexityCalc
        elif method == "entropy":
            self.calc_cls = EntropyComplexityCalc
        else:
            raise ValueError("Invalid method. Choose 'bdm' or 'entropy'.")

        assert strategy in {"min_increase", "neutral_or_decrease", "max_decrease"}, \
            "Invalid strategy. Choose from 'min_increase', 'neutral_or_decrease', or 'max_decrease'."
        self.strategy = strategy

    def prune(self, model: nn.Module, n_nodes: int) -> nn.Module:
        model = deepcopy(model)
        layers = get_layer_sequence(model)

        pruned = 0
        while pruned < n_nodes:
            base_complexity = self.calc_cls().compute(model)
            best = None

            if self.strategy == "max_decrease":
                best_delta = float("-inf")
            else:
                best_delta = float("inf")

            for layer_idx in range(len(layers) - 1):  # skip last layer
                curr_weights = layers[layer_idx][1]
                next_weights = layers[layer_idx + 1][1]
                num_rows = curr_weights.shape[0]

                for i in range(num_rows):
                    if torch.all(curr_weights[i] == 0.0):
                        continue

                    # Backup
                    row_backup = curr_weights[i].clone()
                    col_backup = next_weights[:, i].clone()

                    # Simulate pruning
                    curr_weights.data[i] = 0.0
                    next_weights.data[:, i] = 0.0

                    # Compute delta
                    delta = abs(base_complexity - self.calc_cls().compute(model))

                    # Strategy logic
                    if self.strategy == "min_increase":
                        if delta < best_delta:
                            best = (layer_idx, i)
                            best_delta = delta
                    elif self.strategy == "neutral_or_decrease":
                        if abs(delta) < abs(best_delta):
                            best = (layer_idx, i)
                            best_delta = delta
                    elif self.strategy == "max_decrease":
                        if delta > best_delta:
                            best = (layer_idx, i)
                            best_delta = delta

                    # Restore
                    curr_weights.data[i] = row_backup
                    next_weights.data[:, i] = col_backup

            if best is not None:
                layer_idx, i = best
                layers[layer_idx][1].data[i] = 0.0
                layers[layer_idx + 1][1].data[:, i] = 0.0
                pruned += 1
            else:
                break

        return model
