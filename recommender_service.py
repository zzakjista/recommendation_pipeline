import os 
import json
import torch
import bentoml
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from recommender.dataset import data_source_factory, preprocessor_factory, dataset_factory
from recommender.dataloader import dataloader_factory
from recommender.model import model_factory
from recommender.runner import runner_factory
from recommender.generator.autoencoder import AEGenerator
from recommender.util import save_bento_model   

@hydra.main(config_path='recommender/configs', config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    """
    학습을 진행하고 산출물을 저장하는 기능
    :input: 
        - args
    :return:
        - None
    :output:
        - checkpoint/config.json
        - checkpoint/vocab.json
        - checkpoint/{args.model_code}.pth
    """
    hydra_dir = os.getcwd()
    original_dir = get_original_cwd()
    os.chdir(original_dir)
    dataset = dataset_factory(cfg)
    dataloader = dataloader_factory(dataset, cfg)
    model = model_factory(cfg)
    runner = runner_factory(model, dataloader, cfg)
    runner.train()
    return None
    config = args.__dict__
    with open(f'recommender/checkpoint/config.json', 'w') as f:
        json.dump(config, f)
    save_bento_model(runner.model, args)


def inference(request):
    """
    실제 inference가 수행되는 함수 
    :input: 
        - request
    :return:
        - response
    *특이 사항
        - arg, vocab, model의 버전 관리 기능은 미구현됨
        - 다만 앱 서버에서 추천 결과를 return하는 기능은 조회가 가능함
    """
    with open(f'recommender/checkpoint/config.json', 'r') as f:
        config = json.load(f)
    args.__dict__.update(config)
    with open(f'recommender/checkpoint/vocab.json', 'r') as f:
        vocab = json.load(f)
    model = bentoml.pytorch.load_runner(args.model_code)
    model.init_local()
    generator = AEGenerator(model, args, vocab)
    response = generator.generate(0)
    return response

    
if __name__ == '__main__':
    train()


