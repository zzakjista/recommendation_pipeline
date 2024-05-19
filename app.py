from config import *
from dataset import data_source_factory, preprocessor_factory, dataset_factory
from dataloader import dataloader_factory
from model import model_factory
from runner import runner_factory
from arguments import args

def main(args):
    # service_config = ServiceConfig()
    # model_config = ModelConfig()
    data = data_source_factory(args)
    preprocessor_factory(data=data, args=args)
    dataset = dataset_factory(args)
    dataloader = dataloader_factory(dataset, args)
    model = model_factory(args)
    runner = runner_factory(model, dataloader, args)
    runner.train()
    # runner.inference()

if __name__ == '__main__':
    main(args)

