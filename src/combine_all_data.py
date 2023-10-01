import sys
import omegaconf

import pandas as pd

from tqdm import tqdm

from pathlib import Path

# We don't use the hydra.compose api, since we can't use sweeps with that anyways. Instead, we literally build a giant dataframe of all outputs in multirun.

def main():

    if len(sys.argv) != 2:
        print("Usage: python src/get_all_data.py. PATH_TO_ALL_DATA \nThis script does not use hydra; do not pass overrides.")
        sys.exit(1)

    # Where to save the giant dataframe
    save_fn = sys.argv[1]

    # Base config file
    config_fp = "conf/config.yaml"
    cfg = omegaconf.OmegaConf.load(config_fp)

    # Assume we're interested in outputs, but can change to multirun
    root_dir = cfg.filepaths.hydra_run_root
    curves_fn = cfg.filepaths.curves_fn
    leaf_hydra_cfg_fn = ".hydra/config.yaml"

    # Collect all the results of individual training experiments
    experiment_results = []
    print(f"collecting all simulation data from {root_dir}.")
    fns = list(Path(root_dir).rglob(curves_fn))
    for path in tqdm(fns):

        parent = path.parent.absolute()

        # Load full config w/ overrides
        leaf_cfg = omegaconf.OmegaConf.load(parent / leaf_hydra_cfg_fn)

        # Create dataframe
        df = pd.read_csv(parent / curves_fn)

        # Annotate with metadata from config
        df["quantifier"] = leaf_cfg.quantifier.name
        df["num_states"] = leaf_cfg.learning.num_states
        df["max_epochs"] = leaf_cfg.learning.epochs
        df["batch_size"] = leaf_cfg.learning.batch_size
        df["learning_rate"] = leaf_cfg.learning.learning_rate
        df["seed"] = leaf_cfg.seed

        experiment_results.append(df)

    # Concat
    df_all = pd.concat(experiment_results, axis=0, ignore_index=True)

    # Save
    df_all.to_csv(save_fn, index=False)
    print(f"Wrote to {save_fn}.")

if __name__ == "__main__":
    main()