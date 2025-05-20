import torch
import numpy as np
from abc import ABC, abstractmethod
from pybdm import BDM


class ModelComplexityCalc(ABC):
    """
    Abstract interface for model complexity calculators.
    """

    def __init__(self, model):
        self.model = model
        self.matrices = self._get_binarized_weight_matrices(model)

    @abstractmethod
    def compute(self):
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

    def __init__(self, model):
        super().__init__(model)
        self.bdm = BDM(ndim=2)
        self.counters = [self.bdm.decompose_and_count(m) for m in self.matrices]

    def compute(self):
        return self.bdm.compute_bdm(*self.counters)


class EntropyComplexityCalc(ModelComplexityCalc):
    """
    Computes Shannon entropy over all model weight matrices.
    """

    def compute(self):
        total_entropy = 0
        for mat in self.matrices:
            values, counts = np.unique(mat, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs))
            total_entropy += entropy
        return total_entropy
