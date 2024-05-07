from .read_raw_data import DataReader
from .make_interaction_data import AmazonGamesPreprocessor, SteamGamesPreprocessor
from .ae_dataset import AEDataset

def data_source_factory(game_name:str):
    reader = DataReader('amazon', 'games_review', ('2012-01-01', '2012-12-31'))
    if game_name.lower() == 'amazon_games':
        data = reader.read_gz_to_pandas('data/Video_Games_5.json.gz')
    elif game_name.lower() == 'steam_games':
        data = reader.read_csv_to_pandas('data/steam-200k.csv', header=None, names=['user_id', 'item_id', 'action', 'rating', 'dummy'])
    else:
        raise ValueError('Invalid game name')
    return data

def preprocessor_factory(game_name:str, data):
    if game_name.lower() == 'amazon_games':
        preprocesser = AmazonGamesPreprocessor(data)
    elif game_name.lower() == 'steam_games':
        preprocesser = SteamGamesPreprocessor(data)
    else:
        raise ValueError('Invalid data name')
    preprocesser.preprocess()
    preprocesser.save(f'data/{game_name}_interaction_data.pkl')
    return None

def dataset_factory(game_name:str):
    dataset = AEDataset(game_name)
    dataset.make_user_item_dict()
    dataset.train_valid_split()
    return dataset



    