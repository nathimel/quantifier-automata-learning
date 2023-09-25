"""Various helper/utility functions."""

import hydra
import os
import torch

import pandas as pd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_seed(seed: int) -> None:
    """Sets random seeds."""
    torch.manual_seed(seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_curves(curves: list, fn: str) -> None:
    """Save loss, accuracy curves to a csv."""
    df_curves = pd.DataFrame(curves)
    df_curves["epoch"] = df_curves.index + 1
    df_curves.to_csv(fn, index=False)
    print(f"Wrote curves to {os.path.join(os.getcwd(), fn)}.")    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# File handling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_quantifier_data_fn(config):
    return os.path.join(os.path.join(hydra.utils.get_original_cwd(), config.filepaths.quantifier_data_dir), f"{config.learning.data.max_bitstring_length}.csv")
