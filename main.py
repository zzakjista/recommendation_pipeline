from config import *
from dataset.read_raw_data import data_source_factory
from dataset.make_interaction_data import preprocessor_factory
from dataset import dataset_factory
from dataloader import dataloader_factory
from model import model_factory
from runner import runner_factory

def main():
    service_config = ServiceConfig()
    model_config = ModelConfig()

    data = data_source_factory(service_config.game_name)
    preprocessor_factory(service_config.game_name, data)
    dataset = dataset_factory(service_config.game_name)
    dataloader = dataloader_factory(dataset, batch_size=32, shuffle=True, num_workers=0)
    model = model_factory(model_name=model_config.model_name, input_dim=dataset.num_items, hidden_dim=64)
    runner = runner_factory(model, dataloader, model_config.optimizer, model_config.criterion, model_config.lr, model_config.device, dataset, scheduler=None)
    runner.train(10)


if __name__ == '__main__':
    main()

