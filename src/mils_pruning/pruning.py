import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy
from mils_pruning.complexity import BDMComplexityCalc, EntropyComplexityCalc

class Pruner:
    def prune(self, model: nn.Module, n_weights: int) -> nn.Module:
        raise NotImplementedError


class RandomPruner(Pruner):
    def prune(self, model: nn.Module, n_weights: int) -> nn.Module:
        model = deepcopy(model)
        weights = []

        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                flat = param.data.view(-1)
                indices = [(name, i) for i in range(flat.numel())]
                weights.extend(indices)

        to_prune = random.sample(weights, n_weights)
        for (name, idx) in to_prune:
            param = dict(model.named_parameters())[name]
            flat = param.data.view(-1)
            flat[idx] = 0.0

        return model


class MILSPruner(Pruner):
    def __init__(self, method="bdm"):
        if method == "bdm":
            self.calc_cls = BDMComplexityCalc
        elif method == "entropy":
            self.calc_cls = EntropyComplexityCalc
        else:
            raise ValueError("Invalid method. Choose 'bdm' or 'entropy'.")

    def prune(self, model: nn.Module, n_weights: int) -> nn.Module:
        model = deepcopy(model)

        for step in range(n_weights):
            min_delta = float('inf')
            best_weight = None
            base_calc = self.calc_cls(model)
            base_complexity = base_calc.compute()

            for name, param in model.named_parameters():
                if "weight" in name and param.requires_grad:
                    flat = param.data.view(-1)
                    for idx in range(flat.numel()):
                        original = flat[idx].item()
                        flat[idx] = 0.0

                        temp_calc = self.calc_cls(model)
                        delta = temp_calc.compute() - base_complexity

                        if delta < min_delta:
                            min_delta = delta
                            best_weight = (name, idx)

                        flat[idx] = original

            if best_weight is not None:
                name, idx = best_weight
                param = dict(model.named_parameters())[name]
                param.data.view(-1)[idx] = 0.0

        return model
