from typing import List, Dict, Any
import os 
import yaml
import json
import bentoml
import mlflow 
from omegaconf import OmegaConf
from functools import wraps

def save_bento_model(model, args):
    """
    BentoML을 활용하여 모델을 저장하는 함수
    :input: 
        - model
        - args
        - vocab
    :return:
        - None
    :output:
        - BentoService
    """
    if args.model_code == 'autoencoder':
        model_tag = bentoml.pytorch.save_model(args.model_code, model)
    elif args.model_code == 'ease':
        model_tag = bentoml.picklable.save_model(args.model_code, model)
    else:
        raise ValueError(f'Not supported model_code: {args.model_code}')
    return model_tag
    
def mlflow_run_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # MLflow 세션 시작
        mlflow.start_run()
        try:
            result = func(*args, **kwargs)
        finally:
            # MLflow 세션 종료
            mlflow.end_run()
        return result
    return wrapper

def complete_experiment(config:OmegaConf, vocab:Dict, model:Any)-> None:
    config = OmegaConf.to_yaml(config, resolve=True)
    with open('config.yaml', 'w') as f:
        f.write(config)
    mlflow.log_artifact('config.yaml')
    os.remove('config.yaml')
    mlflow.log_dict(vocab, 'vocab.json')
    mlflow.pytorch.log_model(model, "model")
    return None


def load_artifact(experiment_name:str):
    run_id = search_latest_run_id(experiment_name)
    destination_path = "artifacts"
    config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.yaml", dst_path=destination_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = OmegaConf.create(config)

    vocab_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="vocab.json", dst_path=destination_path)
    with open(vocab_path, "r") as file:
        vocab = json.load(file)

    model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=destination_path)
    model = mlflow.pytorch.load_model(model_path)

    return config, vocab, model

def search_latest_run_id(experiment_name:str):
    runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time desc"], max_results=1)
    return runs['run_id'][0]