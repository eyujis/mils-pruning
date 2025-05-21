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

    Returns:
        List of (name, param) tuples for all prunable 2D linear layers.
    """
    return [
        (name, param)
        for name, param in model.named_parameters()
        if "weight" in name and param.requires_grad and param.ndim == 2
    ]


class RandomPruner(Pruner):
    """
    Randomly removes `n_nodes` neurons by zeroing:
    - the row of the current layer (outgoing weights)
    - the corresponding column in the next layer (incoming weights)
    """
    def __init__(self):
        self.level = "node"

    def prune(self, model: nn.Module, n_nodes: int) -> nn.Module:
        model = deepcopy(model)
        layers = get_layer_sequence(model)

        pruned = 0
        while pruned < n_nodes:
            candidates = []
            for layer_idx in range(len(layers) - 1):
                param = layers[layer_idx][1]
                for i in range(param.shape[0]):
                    if not torch.all(param[i] == 0.0):
                        candidates.append((layer_idx, i))

            if not candidates:
                break

            layer_idx, i = random.choice(candidates)
            curr_param = layers[layer_idx][1]
            next_param = layers[layer_idx + 1][1]

            curr_param.data[i] = 0.0
            next_param.data[:, i] = 0.0
            pruned += 1

        return model



class MILSPruner(Pruner):
    """
    MILS-based neuron pruning with configurable complexity-based strategies.

    Strategies:
        - 'min_increase'   : prune neuron that causes the smallest *positive* information contribution.
                             Falls back to the smallest absolute delta if none are positive.
        - 'max_increase'   : prune neuron that causes the largest increase in complexity.
        - 'min_absolute'   : prune neuron with the smallest absolute change in complexity.
        - 'max_absolute'   : prune neuron with the largest absolute change in complexity.
        - 'min_decrease'   : prune neuron with the smallest *negative* delta (closest to zero).
                             Falls back to the most negative delta if no negatives exist.
        - 'max_decrease'   : prune neuron that causes the largest decrease in complexity.
    """
    def __init__(self, method="bdm", strategy="min_absolute"):
        if method == "bdm":
            self.calc_cls = BDMComplexityCalc
        elif method == "entropy":
            self.calc_cls = EntropyComplexityCalc
        else:
            raise ValueError("Invalid method. Choose 'bdm' or 'entropy'.")

        allowed = {
            "min_increase", "max_increase",
            "min_absolute", "max_absolute",
            "min_decrease", "max_decrease"
        }
        assert strategy in allowed, f"Invalid strategy. Choose from {allowed}."
        self.strategy = strategy
        self.level = "node"

    def prune(self, model: nn.Module, n_nodes: int) -> nn.Module:
        model = deepcopy(model)
        layers = get_layer_sequence(model)

        pruned = 0
        while pruned < n_nodes:
            base_complexity = self.calc_cls().compute(model, mode="node")
            candidates = []

            for layer_idx in range(len(layers) - 1):
                curr_weights = layers[layer_idx][1]
                next_weights = layers[layer_idx + 1][1]

                for i in range(curr_weights.shape[0]):
                    if torch.all(curr_weights[i] == 0.0):
                        continue

                    row_backup = curr_weights[i].clone()
                    col_backup = next_weights[:, i].clone()

                    curr_weights.data[i] = 0.0
                    next_weights.data[:, i] = 0.0

                    after_complexity = self.calc_cls().compute(model, mode="node")
                    delta = base_complexity - after_complexity

                    candidates.append(((layer_idx, i), delta))

                    curr_weights.data[i] = row_backup
                    next_weights.data[:, i] = col_backup

            if not candidates:
                break

            if self.strategy == "min_increase":
                pos = [(c, d) for c, d in candidates if d > 0]
                best = min(pos, key=lambda x: x[1])[0] if pos else min(candidates, key=lambda x: abs(x[1]))[0]

            elif self.strategy == "max_increase":
                best = max(candidates, key=lambda x: x[1])[0]

            elif self.strategy == "min_absolute":
                best = min(candidates, key=lambda x: abs(x[1]))[0]

            elif self.strategy == "max_absolute":
                best = max(candidates, key=lambda x: abs(x[1]))[0]

            elif self.strategy == "min_decrease":
                neg = [(c, d) for c, d in candidates if d < 0]
                best = max(neg, key=lambda x: x[1])[0] if neg else min(candidates, key=lambda x: x[1])[0]

            elif self.strategy == "max_decrease":
                best = min(candidates, key=lambda x: x[1])[0]

            if best is not None:
                layer_idx, i = best
                layers[layer_idx][1].data[i] = 0.0
                layers[layer_idx + 1][1].data[:, i] = 0.0
                pruned += 1
            else:
                break

        return model


class WeightMILSPruner(Pruner):
    """
    MILS-based weight pruning using BDM with polarity-specific binary views.

    Polarity-aware pruning prioritizes the currently dominant polarity in the network.
    For each step, it selects weights with the polarity (+1 or -1) that is more frequent.

    Strategies:
        - 'min_increase', 'max_increase'
        - 'min_absolute', 'max_absolute'
        - 'min_decrease', 'max_decrease'
    """
    def __init__(self, method="bdm", strategy="min_absolute"):
        if method != "bdm":
            raise ValueError("Only BDM is currently supported for weight-level MILS.")

        self.calc_cls = BDMComplexityCalc

        allowed = {
            "min_increase", "max_increase",
            "min_absolute", "max_absolute",
            "min_decrease", "max_decrease"
        }
        assert strategy in allowed, f"Invalid strategy. Choose from {allowed}."
        self.strategy = strategy
        self.level = "weight"

    def prune(self, model: nn.Module, n_weights: int) -> nn.Module:
        model = deepcopy(model)
        layers = get_layer_sequence(model)
        pruned = 0

        while pruned < n_weights:
            candidates = []

            # Count polarity distribution
            current_counts = {+1: 0, -1: 0}
            for name, param in model.named_parameters():
                if "weight" in name and param.ndim == 2:
                    current_counts[+1] += (param == 1).sum().item()
                    current_counts[-1] += (param == -1).sum().item()

            target_polarity = +1 if current_counts[+1] >= current_counts[-1] else -1

            for layer_idx, (name, param) in enumerate(layers):
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        w = param[i, j].item()
                        if w == 0.0:
                            continue

                        polarity = +1 if w > 0 else -1
                        if polarity != target_polarity:
                            continue

                        original = w
                        base = self.calc_cls().compute(model, mode="weight", polarity=polarity)
                        param.data[i, j] = 0.0
                        after = self.calc_cls().compute(model, mode="weight", polarity=polarity)
                        delta = base - after
                        param.data[i, j] = original

                        candidates.append(((layer_idx, i, j, polarity), delta))

            if not candidates:
                break

            if self.strategy == "min_increase":
                pos = [(c, d) for c, d in candidates if d > 0]
                best = min(pos, key=lambda x: x[1])[0] if pos else min(candidates, key=lambda x: abs(x[1]))[0]

            elif self.strategy == "max_increase":
                best = max(candidates, key=lambda x: x[1])[0]

            elif self.strategy == "min_absolute":
                best = min(candidates, key=lambda x: abs(x[1]))[0]

            elif self.strategy == "max_absolute":
                best = max(candidates, key=lambda x: abs(x[1]))[0]

            elif self.strategy == "min_decrease":
                neg = [(c, d) for c, d in candidates if d < 0]
                best = max(neg, key=lambda x: x[1])[0] if neg else min(candidates, key=lambda x: x[1])[0]

            elif self.strategy == "max_decrease":
                best = min(candidates, key=lambda x: x[1])[0]

            if best is not None:
                layer_idx, i, j, polarity = best
                layers[layer_idx][1].data[i, j] = 0.0
                pruned += 1
            else:
                break

        return model
    

class WeightRandomPruner(Pruner):
    """Randomly removes individual nonzero weights from the model."""
    def __init__(self):
        self.level = "weight"

    def prune(self, model: nn.Module, n_weights: int) -> nn.Module:
        model = deepcopy(model)
        weights = []

        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad and param.ndim == 2:
                indices = (param != 0).nonzero(as_tuple=False)
                for idx in indices:
                    i, j = idx.tolist()
                    weights.append((name, i, j))

        to_prune = random.sample(weights, min(n_weights, len(weights)))

        for (name, i, j) in to_prune:
            param = dict(model.named_parameters())[name]
            param.data[i, j] = 0.0

        return model

