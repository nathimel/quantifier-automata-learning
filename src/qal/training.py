"""Various training-specific helpers, wrappers"""

import torch


def get_optimizer(name: str) -> torch.optim.Optimizer:
    return {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }[name]

