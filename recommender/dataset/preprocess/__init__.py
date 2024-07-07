from .interaction_dataset import AmazonGamesPreprocessor, SteamGamesPreprocessor, MovieLensPreprocessor
from .model_dataset import AEDataset, EASEDataset

def preprocessor_factory(data, args):
    if args.data_code.lower() == 'amazon_games':
        preprocesser = AmazonGamesPreprocessor(data, args)
    elif args.data_code.lower() == 'steam_games':
        preprocesser = SteamGamesPreprocessor(data, args)
    elif args.data_code.lower() == 'ml-100k':
        preprocesser = MovieLensPreprocessor(data, args)
    else:
        raise ValueError('Invalid data name')
    preprocesser.preprocess()
    preprocesser.save(f'recommender/data/{args.data_code}_interaction_data.pkl')
    return None

def model_dataset_factory(args):
    if args.model_code.lower() == 'autoencoder':
        return AEDataset(args)
    elif args.model_code.lower() == 'ease':
        return EASEDataset(args)
    elif args.model_code.lower() == 'autoencoder_si':
        return AEDataset(args)
    elif args.model_code.lower() == 'vae':
        return AEDataset(args)
    else:
        raise ValueError('Invalid model name')