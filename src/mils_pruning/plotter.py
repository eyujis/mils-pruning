import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
from mils_pruning import get_result_file


def load_all_runs(arch_tag, level, name_prefix, num_runs, prune_step):
    acc_matrix, activity_matrix = [], []
    for i in range(num_runs):
        run_name = f"{name_prefix}_run{i}"
        acc = np.load(get_result_file(arch_tag, level, run_name, "accs", prune_step))
        activity = np.load(get_result_file(arch_tag, level, run_name, level + "s", prune_step))
        acc_matrix.append(acc)
        activity_matrix.append(activity)
    return np.array(acc_matrix), np.array(activity_matrix)


def plot_averaged_results(arch_tag, level, names, num_runs, prune_step, sigma=1, show_ci=True):
    """
    Plot averaged accuracy curves (with optional 95% confidence interval) for a given level.

    Parameters
    ----------
    arch_tag : str
        e.g., "arch_32_32"
    level : str
        "node" or "weight"
    names : list of str
        List of experiment name prefixes (without "_runX").
    num_runs : int
        Number of runs per experiment.
    prune_step : int
        Final number of remaining nodes or weights (depends on level).
    sigma : float
        Smoothing factor for accuracy curve.
    show_ci : bool
        Whether to show shaded 95% confidence intervals.
    """
    assert level in {"node", "weight"}, f"Invalid level: {level}"
    suffix = "Nodes" if level == "node" else "Weights"

    plt.figure(figsize=(10, 6))

    for name in names:
        accs, activity = load_all_runs(arch_tag, level, name, num_runs, prune_step)

        mean = accs.mean(axis=0)
        std = accs.std(axis=0)
        ci95 = 1.96 * std / np.sqrt(num_runs)

        mean_smooth = gaussian_filter1d(mean, sigma=sigma)
        ci95_smooth = gaussian_filter1d(ci95, sigma=sigma)

        plt.plot(activity[0], mean_smooth, label=name, linewidth=3)

        if show_ci:
            plt.fill_between(activity[0], mean_smooth - ci95_smooth, mean_smooth + ci95_smooth, alpha=0.2)

    plt.xlabel(f"Active {suffix}")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Average Accuracy vs Pruned {suffix}" + (" (with 95% CI)" if show_ci else ""))
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_single_run(arch_tag, level, names, run_idx, prune_step):
    """
    Plot raw accuracy trajectories for multiple strategies for a single run.

    Parameters
    ----------
    arch_tag : str
        Architecture tag (e.g., "arch_32_32")
    level : str
        Pruning level: "node" or "weight"
    names : list of str
        List of experiment prefixes (e.g., ["mils_min_increase", "random"])
    run_idx : int
        The index of the run to plot (e.g., 0, 1, ...)
    prune_step : int
        The final number of remaining nodes or weights (depends on level)
    """
    assert level in {"node", "weight"}, f"Invalid level: {level}"
    suffix = "Nodes" if level == "node" else "Weights"

    plt.figure(figsize=(10, 6))

    for name in names:
        run_name = f"{name}_run{run_idx}"
        acc = np.load(get_result_file(arch_tag, level, run_name, "accs", prune_step))
        activity = np.load(get_result_file(arch_tag, level, run_name, level + "s", prune_step))
        plt.plot(activity, acc, label=name, linewidth=2)

    plt.xlabel(f"Active {suffix}")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Individual Run {run_idx} — {suffix}")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
