import torch
from mils_pruning import run_pruning_experiment
from mils_pruning.model import BinarizedMLP
from mils_pruning.pruning import WeightMILSPruner, WeightRandomPruner
from mils_pruning.paths import get_model_path, get_results_dir
from mils_pruning.pruning import SelectiveRandomPruner



def run_weight_pruning_experiments(
    arch_tag: str,
    num_runs: int,
    max_removal_ratio: float,
    prune_step: int,
    strategies: list[str],
    test_loader,
    device
):
    nodes_h1, nodes_h2 = map(int, arch_tag.replace("arch_", "").split("_"))

    for run_idx in range(num_runs):
        model_path = get_model_path(arch_tag, run_idx)

        # for strategy in strategies:
        #     torch.manual_seed(run_idx)
        #     model = BinarizedMLP((10, 10), nodes_h1, nodes_h2).to(device)
        #     model.load_state_dict(torch.load(model_path))

        #     pruner = WeightMILSPruner(method="bdm", strategy=strategy)
        #     results_dir = get_results_dir(arch_tag, pruner.level, prune_step)
        #     results_dir.mkdir(parents=True, exist_ok=True)

        #     run_pruning_experiment(
        #         pruner,
        #         model,
        #         test_loader,
        #         device,
        #         max_removal_ratio=max_removal_ratio,
        #         prune_step=prune_step,
        #         experiment_name=f"mils_{strategy}_run{run_idx}",
        #     )

        # # Random baseline
        # torch.manual_seed(run_idx)
        # model = BinarizedMLP((10, 10), nodes_h1, nodes_h2).to(device)
        # model.load_state_dict(torch.load(model_path))

        # pruner = WeightRandomPruner()
        # results_dir = get_results_dir(arch_tag, pruner.level, prune_step)
        # results_dir.mkdir(parents=True, exist_ok=True)

        # run_pruning_experiment(
        #     pruner,
        #     model,
        #     test_loader,
        #     device,
        #     max_removal_ratio=max_removal_ratio,
        #     prune_step=prune_step,
        #     experiment_name=f"random_run{run_idx}",
        # )

        # Selective random pruning (excluding neutral weights)
        for mode, value in [("percent", 0.05)]:
            torch.manual_seed(run_idx)
            model = BinarizedMLP((10, 10), nodes_h1, nodes_h2).to(device)
            model.load_state_dict(torch.load(model_path))

            pruner = SelectiveRandomPruner(exclude_mode=mode, exclude_value=value)
            results_dir = get_results_dir(arch_tag, pruner.level, prune_step)
            results_dir.mkdir(parents=True, exist_ok=True)

            tag = f"selective_random_{mode}_{value}".replace(".", "")
            run_pruning_experiment(
                pruner,
                model,
                test_loader,
                device,
                max_removal_ratio=max_removal_ratio,
                prune_step=prune_step,
                experiment_name=f"{tag}_run{run_idx}",
            )
