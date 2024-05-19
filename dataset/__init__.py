from .read_raw_data import DataReader
from .make_interaction_data import AmazonGamesPreprocessor, SteamGamesPreprocessor
from .model_dataset import AEDataset, EASEDataset

def data_source_factory(args):
    reader = DataReader('amazon', 'games_review', ('2012-01-01', '2012-12-31')) # DB 객체 생성
    if args.game_name.lower() == 'amazon_games':
        data = reader.read_gz_to_pandas('data/Video_Games_5.json.gz')
    elif args.game_name.lower() == 'steam_games':
        data = reader.read_csv_to_pandas('data/steam-200k.csv', header=None, names=['user_id', 'item_id', 'action', 'rating', 'dummy'])
    else:
        raise ValueError('Invalid game name')
    return data

def preprocessor_factory(data, args):
    if args.game_name.lower() == 'amazon_games':
        preprocesser = AmazonGamesPreprocessor(data)
    elif args.game_name.lower() == 'steam_games':
        preprocesser = SteamGamesPreprocessor(data)
    else:
        raise ValueError('Invalid data name')
    preprocesser.preprocess()
    preprocesser.save(f'data/{args.game_name}_interaction_data.pkl')
    return None

def dataset_factory(args):
    if args.model == 'autoencoder':
        dataset = AEDataset(args)
        dataset.train_valid_split()
        return dataset
    elif args.model == 'ease':
        dataset = EASEDataset(args)
        dataset.train_valid_split()
        return dataset
    else:
        raise ValueError(f"model {args.model} is not supported")



    