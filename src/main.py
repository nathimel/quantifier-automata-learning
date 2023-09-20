"""Main experiment script that trains and evaluates pfa learning."""

import hydra
import util
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)


if __name__ == "__main__":
    main()