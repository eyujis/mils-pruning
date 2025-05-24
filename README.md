# MILS-Based Neural Network Node Pruning

This project implements node-level pruning for binarized neural networks based on the principle of minimal information loss, inspired by the MILS algorithm (Zenil et al., 2024). We use algorithmic complexity (BDM) and Shannon entropy as criteria to guide which neurons to remove during iterative pruning.

## 🔍 Overview

We compare three strategies for pruning hidden-layer neurons:

1. **MILS using BDM** — removes neurons whose removal causes the smallest increase in algorithmic complexity (measured using the Block Decomposition Method).
2. **MILS using Entropy** — removes neurons that minimally change total entropy, using BDM’s decomposition.
3. **Random** — randomly selects and removes neurons for comparison.

At each step, a single neuron is removed by zeroing its outgoing and incoming weights. Complexity is recomputed for the entire network at each step, simulating the effect of each candidate removal.

## ⚙️ Implementation Highlights

- **Binarized MLP**: We use a fully binarized neural network (with ±1 weights and activations) built using PyTorch.
- **Pruning Granularity**: Neurons are pruned one by one, including all their associated edges (weights).
- **MILS Strategy Options**: You can choose to remove:
  - the neuron causing **minimal absolute change** (default, as per MILS definition),
  - the neuron causing **maximal increase** in complexity,
  - or the one causing the **most neutral change** (closest to zero, including possible decreases).

## 📁 Project Structure

```
mils-pruning/
├── notebooks/ # Jupyter experiments
├── results/ # Saved accuracies, nodes, and plots
├── saved_weights/ # Best model checkpoints
├── data/ # MNIST data (downloaded automatically)
├── src/
│ └── mils_pruning/
│ ├── complexity.py # Complexity calculators (BDM & entropy)
│ ├── pruning.py # Node pruning implementations
│ ├── eval.py # Test function
│ ├── training.py # Early stopping + training
│ ├── data.py # MNIST loader with stratified subsets
│ ├── model.py # Binarized MLP with STE
│ ├── config.py # Paths and data locations
│ └── experiment_runner.py # Pruning loop & result logging
└── pyproject.toml # Poetry environment config
```

## 🧪 Running Experiments

```python
from mils_pruning.model import BinarizedMLP
from mils_pruning.data import get_mnist_data_loaders
from mils_pruning.pruning import MILSPruner
from mils_pruning.experiment_runner import run_pruning_experiment

train_loaders, val_loader, test_loader = get_mnist_data_loaders()
model = BinarizedMLP(input_shape=(10, 10), nodes_h1=32, nodes_h2=16)

pruner = MILSPruner(method="bdm", strategy="min_increase")  # or "max_increase", "neutral_or_decrease"

run_pruning_experiment(
    pruner,
    model,
    test_loader,
    device="cuda",
    max_removal_ratio=0.5,
    prune_step=1,
    experiment_name="bdm_mils"
)


## 📈 Plotting Results

```python
from plot_results import plot_pruning_results
plot_pruning_results(["bdm_mils", "entropy_mils", "random"])
```

## ⚙️ Requirements

- Python 3.11+
- PyTorch
- NumPy
- pybdm
- scikit-learn
- matplotlib
- tqdm

Install with Poetry:

```bash
poetry install
```

## 🧠 Credits

- Based on MILS: Zenil, H., Kiani, N. A., Adams, A., Abrahão, F. S., Rueda-Toicen, A., Zea, A. A., & Tegnér, J. (2018). Minimal algorithmic information loss methods for dimension reduction, feature selection and network sparsification. arXiv preprint [arXiv:1802.05843](https://arxiv.org/abs/1802.05843).
- Complexity via [pybdm](https://pypi.org/project/pybdm/)

---

For questions or contributions, open an issue or contact the author.
