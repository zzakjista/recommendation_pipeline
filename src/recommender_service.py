import os 
import json
import torch
import bentoml
import argparse
import mlflow 
import hydra
from hydra import compose, initialize
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from recommender.dataset import data_source_factory, preprocessor_factory, dataset_factory
from recommender.dataloader import dataloader_factory
from recommender.model import model_factory
from recommender.runner import runner_factory
from recommender.generator.autoencoder import AEGenerator
from recommender.utils import save_bento_model, mlflow_run_decorator, load_artifact, complete_experiment


@mlflow_run_decorator
# @hydra.main(config_path='recommender/configs', config_name="config.yaml")
def train() -> None:
    with initialize(config_path='recommender/configs'):
        cfg = compose(config_name='config.yaml')
    dataset = dataset_factory(cfg)
    dataloader = dataloader_factory(dataset, cfg)
    model = model_factory(cfg)
    runner = runner_factory(model, dataloader, cfg)
    runner.train()
    complete_experiment(cfg, dataset.vocab, runner.model)
    
def load_generator():
    config, vocab, model = load_artifact('my_experiment')
    generator = AEGenerator(model, config, vocab)
    return generator

def inference(generator, request):
    """
    실제 inference가 수행되는 함수 
    :input:
        - generator 
        - request
    :return:
        - response
    *특이 사항
        - arg, vocab, model의 버전 관리 기능은 미구현됨
        - 다만 앱 서버에서 추천 결과를 return하는 기능은 조회가 가능함
    """
    response = generator.generate(0)
    return response



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for recommend model')   
    parser.add_argument("--experiment_name", type=str, default='v1', help='none')
    args = parser.parse_args()
    mlflow.set_experiment(args.experiment_name)
    train()



