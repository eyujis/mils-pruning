from pathlib import Path

def get_model_path(arch_tag: str, run_idx: int) -> Path:
    """
    Returns the path to the trained model for a given architecture and run index.
    Example: saved_weights/arch_32_32/run0/best_model.pt
    """
    return Path("saved_weights") / arch_tag / f"run{run_idx}" / "best_model.pt"


def get_results_dir(arch_tag: str, level: str) -> Path:
    """
    Returns the base results directory for a given architecture and pruning level.
    Example: results/arch_32_32/nodes/ or results/arch_32_32/weights/
    
    Parameters
    ----------
    arch_tag : str
        Architecture tag, e.g. "arch_32_32"
    level : str
        Pruning level: must be "node" or "weight"
    """
    level_map = {"node": "nodes", "weight": "weights"}
    assert level in level_map, f"Invalid level: {level}. Must be 'node' or 'weight'."
    return Path("results") / arch_tag / level_map[level]


def get_result_file(arch_tag: str, level: str, experiment_name: str, suffix: str) -> Path:
    """
    Returns the full path to a specific result file.
    Example: results/arch_32_32/nodes/mils_min_absolute_run0_accs.npy

    Parameters
    ----------
    arch_tag : str
    level : str
    experiment_name : str
        Prefix for the experiment (e.g., 'mils_min_absolute_run0')
    suffix : str
        Either 'accs', 'nodes', or 'weights'
    """
    return get_results_dir(arch_tag, level) / f"{experiment_name}_{suffix}.npy"
