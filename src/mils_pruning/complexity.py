import torch
import numpy as np
from abc import ABC, abstractmethod
from pybdm import BDM


class ModelComplexityCalc(ABC):
    """
    Abstract interface for model complexity calculators.
    """

    @abstractmethod
    def compute(self, model):
        pass

    def _get_binarized_weight_matrices(self, model):
        matrices = []
        for key, weights in model.state_dict().items():
            if "weight" in key and "bn" not in key.lower():
                signed = weights.sign()
                binarized = torch.where(signed == -1, torch.tensor(0, dtype=torch.int), signed)
                matrices.append(binarized.cpu().numpy().astype(np.int32))
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
