#hydra로 읽기
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from hydra import compose, initialize
from hydra.utils import instantiate



@hydra.main(config_path='../recommender/configs', config_name="config.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg)
    optim = instantiate(cfg.optimizer, lr=cfg.optimizer.lr)
    print(optim)
    print(type(optim))


if __name__ == "__main__":
    my_app()
    

    # override config
    # with initialize(version_base=None, config_path="../recommender/configs", job_name="test_app"):
    #     cfg = compose(config_name="config", overrides=["dataset.num_items=100"])
    # print(OmegaConf.to_yaml(cfg))
