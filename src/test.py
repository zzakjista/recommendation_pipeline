# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import omegaconf
from omegaconf import DictConfig

import hydra
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./recommender/configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(f"Output_dir={HydraConfig.get().runtime.output_dir}")
    # numitem 업데이트
    cfg.dataset.num_items = 100
    log.info(f"num_items={cfg.dataset.num_items}")
    # config local 지정 경로에 저장
    with open("config.yaml", "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg))

def parser_test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", type=int)
    args = parser.parse_args()
    print(args.foo)


if __name__ == "__main__":
    parser_test()