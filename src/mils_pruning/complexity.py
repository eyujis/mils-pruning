import torch
import numpy as np
from abc import ABC, abstractmethod
from pybdm import BDM
from torch import nn


class ModelComplexityCalc(ABC):
    """
    Abstract interface for model complexity calculators.
    """

    @abstractmethod
    def compute(self, model):
        pass

    def _get_binarized_weight_matrices(self, model, mode="node", polarity=None):
        matrices = []

        for name, param in model.named_parameters():
            if "weight" in name and param.ndim == 2:
                signed = param.detach().cpu().sign()

                if mode == "node":
                    # Binarize: -1 → 0, +1 → 1
                    binarized = torch.where(signed == -1, 0, signed)

                elif mode == "weight":
                    assert polarity in {-1, +1}, "Polarity must be -1 or +1 for weight mode"
                    binarized = (signed == polarity).int()  # only +1 or -1 → 1, rest → 0

                else:
                    raise ValueError(f"Invalid mode: {mode}")

                matrices.append(binarized.numpy().astype(np.int32))

        return matrices


class BDMComplexityCalc(ModelComplexityCalc):
    """
    Computes BDM complexity using pybdm.
    """

    def __init__(self):
        self.bdm = BDM(ndim=2)

    def compute(self, model):
        matrices = self._get_binarized_weight_matrices(model)
        counters = [self.bdm.decompose_and_count(m) for m in matrices]
        return self.bdm.compute_bdm(*counters)
    
    def compute(self, model, mode="node", polarity=None):
        matrices = self._get_binarized_weight_matrices(model, mode, polarity)
        counters = [self.bdm.decompose_and_count(m) for m in matrices]
        return self.bdm.compute_bdm(*counters)



class EntropyComplexityCalc(ModelComplexityCalc):
    """
    Computes Shannon entropy using pybdm's block decomposition.
    """

    def __init__(self):
        self.bdm = BDM(ndim=2)

    def compute(self, model):
        matrices = self._get_binarized_weight_matrices(model)
        counters = [self.bdm.decompose_and_count(m) for m in matrices]
        return self.bdm.compute_ent(*counters)

def binarize_weights_for_bdm(weights: torch.Tensor, polarity: int) -> np.ndarray:
    """
    Converts binarized weights to a binary matrix for BDM analysis:
    - polarity = +1: maps +1 → 1, others → 0
    - polarity = -1: maps -1 → 1, others → 0
    """
    assert polarity in [+1, -1]
    tensor = weights.detach().cpu().numpy()
    if polarity == +1:
        return (tensor == 1).astype(np.uint8)
    else:
        return (tensor == -1).astype(np.uint8)
