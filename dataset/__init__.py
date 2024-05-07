from .ae_dataset import AEDataset

def dataset_factory(game_name:str):
    dataset = AEDataset(game_name)
    dataset.make_user_item_dict()
    dataset.train_valid_split()
    return dataset
    