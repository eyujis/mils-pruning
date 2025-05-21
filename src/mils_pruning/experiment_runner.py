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
    experiment_name="experiment",
    output_dir=Path("results")
):
    """
    Runs an iterative pruning experiment and saves accuracy/activity over steps.
    """

    model = model.to(device)
    accs = []      # Accuracy at each step
    activity = []  # Active node or weight count at each step

    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure results dir exists

    # --- Pruning level ---
    level = getattr(pruner, "level", "node")
    assert level in {"node", "weight"}, f"Unknown pruning level: {level}"

    # --- Setup total units and counter ---
    if level == "node":
        total = sum(p.shape[0] for n, p in model.named_parameters()
                    if "weight" in n and p.requires_grad and p.ndim == 2)

        def count_active():
            return sum(
                torch.any(p != 0, dim=1).sum().item()
                for n, p in model.named_parameters()
                if "weight" in n and p.requires_grad and p.ndim == 2 and p.shape[0] > 1
            )

    else:  # level == "weight"
        total = sum(p.numel() for n, p in model.named_parameters()
                    if "weight" in n and p.requires_grad and p.ndim == 2)

        def count_active():
            return sum(
                (p != 0).sum().item()
                for n, p in model.named_parameters()
                if "weight" in n and p.requires_grad and p.ndim == 2
            )

    # --- Pruning schedule ---
    target_remaining = int(total * (1 - max_removal_ratio))
    steps = (total - target_remaining) // prune_step

    # --- Initial evaluation ---
    model.eval()
    initial_active = count_active()
    accs.append(test(model, test_loader, device))
    activity.append(initial_active)

    previous_active = initial_active

    # --- Pruning loop ---
    for step in trange(steps, desc=f"Pruning ({experiment_name})"):

        if level == "weight":
            model = pruner.prune(model, n_weights=prune_step)
        else:
            model = pruner.prune(model, n_nodes=prune_step)

        current_active = count_active()

        if current_active >= previous_active:
            print(f"[Warning] Active {level}s did not decrease at step {step}: "
                  f"{previous_active} → {current_active}")

        assert current_active < previous_active, (
            f"[Error] Active {level}s should strictly decrease at step {step}. "
            f"Previous: {previous_active}, Current: {current_active}"
        )

        previous_active = current_active

        model.eval()
        acc = test(model, test_loader, device)
        accs.append(acc)
        activity.append(current_active)

    # --- Save results ---
    suffix = "weights" if level == "weight" else "nodes"
    np.save(output_dir / f"{experiment_name}_accs.npy", np.array(accs))
    np.save(output_dir / f"{experiment_name}_{suffix}.npy", np.array(activity))
