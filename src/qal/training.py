"""Various training-specific helpers, wrappers"""

import torch


def get_optimizer(name: str) -> torch.optim.Optimizer:
    return {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }[name]

def rand_init(shape: tuple[int], init_temperature: float) -> torch.Tensor:
    """Energy-based initialization from Normal distribution; higher init_temperature -> values are closer to 0."""
    return 1/init_temperature * torch.randn(shape)

