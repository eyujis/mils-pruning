import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from mils_pruning.eval import test

def run_pruning_experiment(pruner, model, test_loader, device, max_removal_ratio=0.5, prune_step=1, experiment_name="experiment"):
    model = model.to(device)
    accs = []
    active_weights = []

    # Total number of weights before pruning
    total_weights = sum(p.numel() for n, p in model.named_parameters() if "weight" in n and p.requires_grad)
    target_remaining = int(total_weights * (1 - max_removal_ratio))
    current_pruned = 0

    # Output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    steps = (total_weights - target_remaining) // prune_step
    for step in trange(steps, desc=f"Pruning ({experiment_name})"):
        model = pruner.prune(model, n_weights=prune_step)
        current_pruned += prune_step

        acc = test(model, test_loader, device)
        accs.append(acc)
        active_weights.append(total_weights - current_pruned)

    accs = np.array(accs)
    weights = np.array(active_weights)

    np.save(output_dir / f"{experiment_name}_accs.npy", accs)
    np.save(output_dir / f"{experiment_name}_weights.npy", weights)
