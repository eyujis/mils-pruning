from pathlib import Path

def get_model_path(arch_tag: str, run_idx: int) -> Path:
    """
    Returns the path to the trained model for a given architecture and run index.
    Example: saved_weights/arch_32_32/run0/best_model.pt
    """
    return Path("saved_weights") / arch_tag / f"run{run_idx}" / "best_model.pt"


def get_results_dir(arch_tag: str, level: str, prune_step: int) -> Path:
    """
    Returns the results directory path for a given architecture, level, and pruning step.
    Example: results/arch_32_32/nodes/prune_step_32/
    """
    level_map = {"node": "nodes", "weight": "weights"}
    assert level in level_map, f"Invalid level: {level}. Must be 'node' or 'weight'."
    return Path("results") / arch_tag / level_map[level] / f"prune_step_{prune_step}"


def get_result_file(
    arch_tag: str,
    level: str,
    experiment_name: str,
    suffix: str,
    prune_step: int
) -> Path:
    """
    Returns the full path to a specific result file under the correct structured directory.
    Example: results/arch_32_32/nodes/prune_step_32/mils_min_absolute_run0_accs.npy
    """
    return get_results_dir(arch_tag, level, prune_step) / f"{experiment_name}_{suffix}.npy"
