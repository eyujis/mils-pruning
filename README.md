# MILS-Based Neural Network Pruning

This project implements minimal-information-loss pruning methods for binarized neural networks, using algorithmic complexity and entropy-based selection criteria. The goal is to study how different complexity measures influence pruning performance and model generalization.

## 🔍 Overview

We evaluate three pruning strategies:

1. **MILS using BDM** — removes weights that contribute least to algorithmic complexity (Block Decomposition Method)
2. **MILS using Entropy** — removes weights that contribute least to entropy, using the same BDM decomposition
3. **Random** — randomly removes weights for comparison

Complexity is computed on binarized versions of the model weights, excluding BatchNorm layers.

## 📁 Project Structure

```
mils-pruning/
├── notebooks/              # Jupyter experiments
├── results/                # Saved accuracies, weights, and plots
├── saved_weights/          # Model checkpoints
├── data/                   # MNIST data (downloaded automatically)
├── src/
│   └── mils_pruning/
│       ├── complexity.py   # Complexity calculators (BDM & entropy)
│       ├── pruning.py      # Pruning methods (MILS & random)
│       ├── eval.py         # Evaluation utilities (test accuracy)
│       ├── training.py     # Model training with early stopping
│       ├── data.py         # Stratified MNIST loader
│       ├── model.py        # Binarized MLP with STE
│       ├── config.py       # Paths
│       └── experiment_runner.py # Iterative pruning & result saving
└── pyproject.toml          # Poetry environment
```

## 🧪 Running Experiments

Train a model and prune it:

```python
from mils_pruning.model import BinarizedMLP
from mils_pruning.pruning import MILSPruner
from mils_pruning.experiment_runner import run_pruning_experiment

model = BinarizedMLP(input_shape=(10, 10), nodes_h1=32, nodes_h2=16)
pruner = MILSPruner(method="bdm")

run_pruning_experiment(
    pruner,
    model,
    test_loader,
    device,
    max_removal_ratio=0.5,
    prune_step=5,
    experiment_name="bdm_mils"
)
```

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

- Based on MILS: Minimal Information Loss Subgraphs (Zenil et al. 2024)
- Complexity via [pybdm](https://pypi.org/project/pybdm/)

---

For questions or contributions, open an issue or contact the author.
