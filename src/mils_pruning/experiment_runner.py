import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from mils_pruning.eval import test
from mils_pruning.paths import get_result_file
import torch


def run_pruning_experiment(
    pruner,
    model,
    test_loader,
    device,
    max_removal_ratio=0.5,
    prune_step=1,
    experiment_name="experiment",
    arch_tag="arch_32_32"
):
    """
    Runs an iterative pruning experiment and saves accuracy/activity over steps
    in a structured results folder based on architecture, level, and prune_step.
    """

    model = model.to(device)
    accs = []      # Accuracy at each step
    activity = []  # Active node or weight count at each step

    # --- Pruning level: "node" or "weight" ---
    level = getattr(pruner, "level", "node")
    assert level in {"node", "weight"}, f"Unknown pruning level: {level}"

    # --- Count total elements based on level ---
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

    # --- Iterative pruning loop ---
    for step in trange(steps, desc=f"Pruning ({experiment_name})"):

        if level == "weight":
            model = pruner.prune(model, n_weights=prune_step)
        else:
            model = pruner.prune(model, n_nodes=prune_step)

        current_active = count_active()

        expected_active = previous_active - prune_step

        if current_active != expected_active:
            print(f"[Warning] Unexpected change in active {level}s at step {step}: "
                f"Expected {expected_active}, but got {current_active} "
                f"(Δ = {previous_active - current_active})")

        assert current_active == expected_active, (
            f"[Error] Pruned count mismatch at step {step}. "
            f"Expected exactly {prune_step} {level}s to be removed. "
            f"Previous: {previous_active}, Current: {current_active}"
        )

        previous_active = current_active
        model.eval()
        accs.append(test(model, test_loader, device))
        activity.append(current_active)

    # --- Save results using correct directory structure ---
    # Note: use the constant prune_step (units removed per iteration), not remaining
    suffix = "weights" if level == "weight" else "nodes"

    # Ensure target folder exists
    output_dir = get_result_file(arch_tag, level, experiment_name, "accs", prune_step).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save accuracy and activity over steps
    np.save(
        get_result_file(arch_tag, level, experiment_name, "accs", prune_step),
        np.array(accs)
    )
    np.save(
        get_result_file(arch_tag, level, experiment_name, suffix, prune_step),
        np.array(activity)
    )

