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
    Returns all linear layers with prunable weights.
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

        candidates = []
        for layer_idx in range(len(layers) - 1):
            param = layers[layer_idx][1]
            for i in range(param.shape[0]):
                if not torch.all(param[i] == 0.0):
                    candidates.append((layer_idx, i))

        to_prune = random.sample(candidates, min(n_nodes, len(candidates)))

        for layer_idx, i in to_prune:
            layers[layer_idx][1].data[i] = 0.0
            layers[layer_idx + 1][1].data[:, i] = 0.0

        return model


class MILSPruner(Pruner):
    """
    MILS-based neuron pruning with configurable complexity-based strategies.

    Strategies:
        - 'min_increase'   : prune neuron with smallest positive contribution.
                             Fallback: smallest absolute delta.
        - 'max_increase'   : prune neuron with largest positive contribution.
        - 'min_absolute'   : prune neuron with smallest absolute delta.
        - 'max_absolute'   : prune neuron with largest absolute delta.
        - 'min_decrease'   : prune neuron with smallest negative delta.
                             Fallback: most negative delta.
        - 'max_decrease'   : prune neuron with largest negative delta.
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
            return model

        if self.strategy == "min_increase":
            pos = [(c, d) for c, d in candidates if d > 0]
            sorted_candidates = sorted(pos, key=lambda x: x[1]) if pos else sorted(candidates, key=lambda x: abs(x[1]))
        elif self.strategy == "max_increase":
            sorted_candidates = sorted(candidates, key=lambda x: -x[1])
        elif self.strategy == "min_absolute":
            sorted_candidates = sorted(candidates, key=lambda x: abs(x[1]))
        elif self.strategy == "max_absolute":
            sorted_candidates = sorted(candidates, key=lambda x: -abs(x[1]))
        elif self.strategy == "min_decrease":
            neg = [(c, d) for c, d in candidates if d < 0]
            sorted_candidates = sorted(neg, key=lambda x: -x[1]) if neg else sorted(candidates, key=lambda x: x[1])
        elif self.strategy == "max_decrease":
            sorted_candidates = sorted(candidates, key=lambda x: x[1])

        to_prune = [c[0] for c in sorted_candidates[:n_nodes]]

        for layer_idx, i in to_prune:
            layers[layer_idx][1].data[i] = 0.0
            layers[layer_idx + 1][1].data[:, i] = 0.0

        return model


class WeightMILSPruner(Pruner):
    """
    MILS-based weight pruning using polarity-specific binary views.

    For each step, selects weights with the dominant polarity (+1 or -1).
    Uses same strategy logic as node-level MILS.
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

        # Count polarity distribution
        counts = {+1: 0, -1: 0}
        for name, param in model.named_parameters():
            if "weight" in name and param.ndim == 2:
                rounded = param.round()
                counts[+1] += (rounded == 1).sum().item()
                counts[-1] += (rounded == -1).sum().item()



        total = counts[+1] + counts[-1]
        if total == 0:
            print("[Warning] No weights to prune.")
            return model  # nothing left to prune

        # Balanced allocation of pruning budget
        n_pos = n_weights // 2
        n_neg = n_weights - n_pos


        pruned = 0
        selected = []

        for polarity, n_target in [(+1, n_pos), (-1, n_neg)]:
            if counts[polarity] == 0 or n_target == 0:
                continue

            base = self.calc_cls().compute(model, mode="weight", polarity=polarity)
            candidates = []

            for layer_idx, (name, param) in enumerate(layers):
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        w = param[i, j].item()
                        if w == 0.0:
                            continue
                        if (w > 0 and polarity != +1) or (w < 0 and polarity != -1):
                            continue

                        original = w
                        param.data[i, j] = 0.0
                        after = self.calc_cls().compute(model, mode="weight", polarity=polarity)
                        delta = base - after
                        param.data[i, j] = original

                        candidates.append(((layer_idx, i, j), delta))

            if not candidates:
                continue

            # Strategy-specific sorting
            if self.strategy == "min_increase":
                pos = [(c, d) for c, d in candidates if d > 0]
                sorted_candidates = sorted(pos, key=lambda x: x[1]) if pos else sorted(candidates, key=lambda x: abs(x[1]))
            elif self.strategy == "max_increase":
                sorted_candidates = sorted(candidates, key=lambda x: -x[1])
            elif self.strategy == "min_absolute":
                sorted_candidates = sorted(candidates, key=lambda x: abs(x[1]))
            elif self.strategy == "max_absolute":
                sorted_candidates = sorted(candidates, key=lambda x: -abs(x[1]))
            elif self.strategy == "min_decrease":
                neg = [(c, d) for c, d in candidates if d < 0]
                sorted_candidates = sorted(neg, key=lambda x: -x[1]) if neg else sorted(candidates, key=lambda x: x[1])
            elif self.strategy == "max_decrease":
                sorted_candidates = sorted(candidates, key=lambda x: x[1])

            selected.extend([c[0] for c in sorted_candidates[:n_target]])

        # Apply all selected prunes
        for layer_idx, i, j in selected:
            layers[layer_idx][1].data[i, j] = 0.0

        return model



class WeightRandomPruner(Pruner):
    """
    Randomly removes `n_weights` individual weights.
    """
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
