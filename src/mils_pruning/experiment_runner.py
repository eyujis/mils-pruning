import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from mils_pruning.eval import test
import torch


def run_pruning_experiment(
    pruner,
    model,
    test_loader,
    device,
    max_removal_ratio=0.5,
    prune_step=1,
    experiment_name="experiment"
):
    model = model.to(device)
    accs = []
    active_nodes = []

    # Total number of neurons (rows in Linear layer weight matrices)
    total_nodes = sum(
        p.shape[0] for n, p in model.named_parameters()
        if "weight" in n and p.requires_grad and p.ndim == 2
    )
    target_remaining = int(total_nodes * (1 - max_removal_ratio))
    current_pruned = 0

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Evaluate original model before pruning
    model.eval()
    initial_active = sum(
        torch.any(p != 0, dim=1).sum().item()
        for n, p in model.named_parameters()
        if "weight" in n and p.requires_grad and p.ndim == 2 and p.shape[0] > 1
    )
    accs.append(test(model, test_loader, device))
    active_nodes.append(initial_active)

    # Begin pruning iterations
    steps = (total_nodes - target_remaining) // prune_step
    for step in trange(steps, desc=f"Pruning ({experiment_name})"):
        model = pruner.prune(model, n_nodes=prune_step)
        current_pruned += prune_step

        active_count = sum(
            torch.any(p != 0, dim=1).sum().item()
            for n, p in model.named_parameters()
            if "weight" in n and p.requires_grad and p.ndim == 2 and p.shape[0] > 1
        )
        expected = total_nodes - current_pruned
        if active_count != expected:
            print(f"[Warning] Expected {expected} active neurons, but found {active_count}!")

        model.eval()
        acc = test(model, test_loader, device)

        accs.append(acc)
        active_nodes.append(active_count)

    # Save results
    np.save(output_dir / f"{experiment_name}_accs.npy", np.array(accs))
    np.save(output_dir / f"{experiment_name}_nodes.npy", np.array(active_nodes))
