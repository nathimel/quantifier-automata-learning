import hydra
import os

from omegaconf import DictConfig

from qal import util

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    print(f"Hydra leaf dir exists or was created successfully at {os.getcwd()}.")

if __name__ == "__main__":
    main()    