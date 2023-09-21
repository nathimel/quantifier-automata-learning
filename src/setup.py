"""Experimental setup script that generates any necessary data before main experiment."""

import hydra
import os
import pandas as pd

from omegaconf import DictConfig
from qal import util
from qal.data_gen import generate_up_to_length, get_quantifier_labeled_data



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    max_length = config.learning.data.max_bitstring_length
    cwd = hydra.utils.get_original_cwd()
    overwrite = config.learning.data.overwrite

    bitstrings_dir = os.path.join(cwd, config.filepaths.bitstrings_dir)

    # Load data if it exists
    data_fn = os.path.join(bitstrings_dir, f"{max_length}.csv")
    if os.path.exists(data_fn) and not overwrite:
        strings = pd.read_csv(data_fn)["string"].astype(int).astype(str).tolist()
        print(f"Loaded {len(strings)} strings from {data_fn}.")
    else:
        # Otherwise generate all strings up to length, then write to disk
        strings = sorted(list(set(generate_up_to_length(max_length))))
        pd.DataFrame(strings, columns=["string"]).to_csv(data_fn, index=False)
        print(f"Wrote {len(strings)} strings to {data_fn}.")

    # Load the quantifier to label data for
    quant_name = config.quantifier.name
    # quantifier_data_fn = os.path.join(os.path.join(cwd, config.filepaths.quantifier_data_dir), f"{max_length}.csv")
    quantifier_data_fn = util.get_quantifier_data_fn(config)

    # breakpoint()

    # Load labeled data if it exists
    if not os.path.exists(quantifier_data_fn) or quant_name not in pd.read_csv(quantifier_data_fn).columns and not overwrite:
        quantifier_data = pd.read_csv(quantifier_data_fn) # reading twice sigh
    else:
        # Otherwise label the data, then write to disk
        strings_df = pd.DataFrame(strings, columns=["string"])
        quantifier_data = get_quantifier_labeled_data(strings_df, quant_name)
        quantifier_data.to_csv(quantifier_data_fn, index=False)
        print(f"Wrote {len(quantifier_data)} labeled strings to {quantifier_data_fn}.")


if __name__ == "__main__":
    main()        