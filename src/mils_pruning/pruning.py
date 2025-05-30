import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy
from mils_pruning.complexity import BDMComplexityCalc, EntropyComplexityCalc
from collections import Counter
from mils_pruning.model import Binarize, BinaryLinear  # or adjust the import path


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
    - 'min_increase'   : prune weight whose removal causes the smallest *positive* increase in complexity.
                            Fallback: prune weight with the smallest absolute change in complexity (|Δ|).
    - 'max_increase'   : prune weight whose removal causes the largest *positive* increase in complexity.
    - 'min_absolute'   : prune weight whose removal causes the smallest absolute change in complexity (|Δ|),
                            regardless of direction.
    - 'max_absolute'   : prune weight whose removal causes the largest absolute change in complexity (|Δ|),
                            regardless of direction.
    - 'min_decrease'   : prune weight whose removal causes the smallest *negative* change in complexity
                            (i.e., smallest simplification).
                            Fallback: prune weight with the smallest absolute change in complexity (|Δ|).
    - 'max_decrease'   : prune weight whose removal causes the largest *negative* change in complexity
                            (i.e., strongest simplification).
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

        # Strategy-specific sorting
        if self.strategy == "min_increase":
            pos = [(c, d) for c, d in candidates if d > 0]

            if len(pos) >= n_nodes:
                sorted_candidates = sorted(pos, key=lambda x: x[1])
            else:
                remaining_needed = n_nodes - len(pos)
                fallback_pool = [cd for cd in candidates if cd not in pos]
                fallback_sorted = sorted(fallback_pool, key=lambda x: abs(x[1]))
                sorted_candidates = sorted(pos, key=lambda x: x[1]) + fallback_sorted[:remaining_needed]

        elif self.strategy == "max_increase":
            sorted_candidates = sorted(candidates, key=lambda x: -x[1])

        elif self.strategy == "min_absolute":
            sorted_candidates = sorted(candidates, key=lambda x: abs(x[1]))

        elif self.strategy == "max_absolute":
            sorted_candidates = sorted(candidates, key=lambda x: -abs(x[1]))

        elif self.strategy == "min_decrease":
            neg = [(c, d) for c, d in candidates if d < 0]

            if len(neg) >= n_nodes:
                sorted_candidates = sorted(neg, key=lambda x: -x[1])  # least negative first
            else:
                remaining_needed = n_nodes - len(neg)
                fallback_pool = [cd for cd in candidates if cd not in neg]
                fallback_sorted = sorted(fallback_pool, key=lambda x: abs(x[1]))
                sorted_candidates = sorted(neg, key=lambda x: -x[1]) + fallback_sorted[:remaining_needed]

        elif self.strategy == "max_decrease":
            sorted_candidates = sorted(candidates, key=lambda x: x[1])  # most negative first

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

    def count_weight_polarities(self, model: torch.nn.Module) -> dict:
        """
        Counts the number of +1 and -1 weights in all BinaryLinear layers
        after applying binarization (STE).
        """
        counts = Counter({+1: 0, -1: 0})

        for name, module in model.named_modules():
            if isinstance(module, BinaryLinear):
                # Apply STE binarization to the weight tensor
                W_b = Binarize.apply(module.weight.data)

                # Count polarity values
                counts[+1] += (W_b == 1).sum().item()
                counts[-1] += (W_b == -1).sum().item()

        return dict(counts)



    def prune(self, model: nn.Module, n_weights: int) -> nn.Module:
        model = deepcopy(model)
        layers = get_layer_sequence(model)

        # Count polarity distribution using binarized weights
        counts = self.count_weight_polarities(model)

        total = counts[+1] + counts[-1]
        if total == 0:
            print("[Warning] No weights to prune.")
            return model  # nothing left to prune

        pos = counts[+1]
        neg = counts[-1]
        total = pos + neg

        # Target: equal remaining +1 and -1 after pruning
        target_each = (total - n_weights) // 2

        # Raw plan: prune down to target_each
        n_pos = max(0, pos - target_each)
        n_neg = max(0, neg - target_each)

        # Ensure total equals n_weights
        total_pruned = n_pos + n_neg
        if total_pruned > n_weights:
            excess = total_pruned - n_weights
            if n_pos >= n_neg:
                n_pos -= excess
            else:
                n_neg -= excess
        elif total_pruned < n_weights:
            deficit = n_weights - total_pruned
            if pos - n_pos > neg - n_neg:
                n_pos += deficit
            else:
                n_neg += deficit

        # Safety check
        assert n_pos + n_neg == n_weights

        selected = []

        for polarity, n_target in [(+1, n_pos), (-1, n_neg)]:
            if counts[polarity] == 0 or n_target == 0:
                continue

            base = self.calc_cls().compute(model, mode="weight", polarity=polarity)
            candidates = []

            for layer_idx, (name, param) in enumerate(layers):
                # Binarize the entire parameter tensor once
                binarized = Binarize.apply(param.data)

                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        if param[i, j].item() == 0.0:
                            continue
                        if binarized[i, j].item() != polarity:
                            continue

                        original = param[i, j].item()
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

                if len(pos) >= n_target:
                    sorted_candidates = sorted(pos, key=lambda x: x[1])
                else:
                    # Fill remainder with smallest absolute deltas from full set (excluding already selected)
                    remaining_needed = n_target - len(pos)
                    fallback_pool = [cd for cd in candidates if cd not in pos]
                    fallback_sorted = sorted(fallback_pool, key=lambda x: abs(x[1]))
                    sorted_candidates = sorted(pos, key=lambda x: x[1]) + fallback_sorted[:remaining_needed]
                    
            elif self.strategy == "max_increase":
                sorted_candidates = sorted(candidates, key=lambda x: -x[1])
            elif self.strategy == "min_absolute":
                sorted_candidates = sorted(candidates, key=lambda x: abs(x[1]))
            elif self.strategy == "max_absolute":
                sorted_candidates = sorted(candidates, key=lambda x: -abs(x[1]))
            elif self.strategy == "min_decrease":
                neg = [(c, d) for c, d in candidates if d < 0]

                if len(neg) >= n_target:
                    sorted_candidates = sorted(neg, key=lambda x: -x[1])  # least negative first
                else:
                    remaining_needed = n_target - len(neg)
                    fallback_pool = [cd for cd in candidates if cd not in neg]
                    fallback_sorted = sorted(fallback_pool, key=lambda x: abs(x[1]))
                    sorted_candidates = sorted(neg, key=lambda x: -x[1]) + fallback_sorted[:remaining_needed]
            
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


class SelectiveRandomPruner(Pruner):
    """
    Randomly prunes weights, excluding the most 'neutral' ones based on absolute delta.

    Modes:
        - exclude_mode = "top_k": exclude a fixed number (exclude_value = 100)
        - exclude_mode = "percent": exclude a fraction of total candidates (e.g., 0.1 for 10%)
    """
    def __init__(self, exclude_mode="percent", exclude_value=0.1, method="bdm"):
        self.level = "weight"
        assert exclude_mode in {"top_k", "percent"}, "exclude_mode must be 'top_k' or 'percent'"
        self.exclude_mode = exclude_mode
        self.exclude_value = exclude_value
        self.calc_cls = BDMComplexityCalc if method == "bdm" else None
        assert self.calc_cls is not None, "Only BDM is currently supported."

    def prune(self, model: torch.nn.Module, n_weights: int) -> torch.nn.Module:
        model = deepcopy(model)
        layers = get_layer_sequence(model)

        candidates = []

        for polarity in [+1, -1]:
            base = self.calc_cls().compute(model, mode="weight", polarity=polarity)

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
            print("[Warning] No eligible weights for pruning.")
            return model

        # Exclude most neutral weights
        sorted_by_neutrality = sorted(candidates, key=lambda x: abs(x[1]))
        if self.exclude_mode == "top_k":
            k = min(self.exclude_value, len(sorted_by_neutrality))
        elif self.exclude_mode == "percent":
            k = int(len(sorted_by_neutrality) * self.exclude_value)

        excluded = set([c[0] for c in sorted_by_neutrality[:k]])
        pool = [c[0] for c in candidates if c[0] not in excluded]

        if len(pool) == 0:
            print("[Warning] No weights left to prune after exclusion.")
            return model

        to_prune = random.sample(pool, min(n_weights, len(pool)))

        for layer_idx, i, j in to_prune:
            layers[layer_idx][1].data[i, j] = 0.0

        return model
