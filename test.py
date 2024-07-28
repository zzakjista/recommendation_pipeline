#hydra로 읽기
import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(config_path='recommender/configs', config_name="config.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.data)


if __name__ == "__main__":
    my_app()
