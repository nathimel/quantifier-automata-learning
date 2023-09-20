"""Experimental setup script that generates any necessary data before main experiment."""

import hydra
import util
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)


if __name__ == "__main__":
    main()
