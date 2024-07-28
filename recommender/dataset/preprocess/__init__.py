from .interaction_dataset import AmazonGamesPreprocessor, SteamGamesPreprocessor, MovieLensPreprocessor
from .model_dataset import AEDataset, EASEDataset

def preprocessor_factory(data, cfg):
    if cfg.data.data_code.lower() == 'amazon_games':
        preprocesser = AmazonGamesPreprocessor(data, cfg)
    elif cfg.data.data_code.lower() == 'steam_games':
        preprocesser = SteamGamesPreprocessor(data, cfg)
    elif cfg.data.data_code.lower() == 'ml-100k':
        preprocesser = MovieLensPreprocessor(data, cfg)
    else:
        raise ValueError('Invalid data name')
    preprocesser.preprocess()
    preprocesser.save(f'recommender/data/{cfg.data.data_code}_interaction_data.pkl')
    return None

def model_dataset_factory(cfg):
    if cfg.model_code.lower() == 'autoencoder':
        return AEDataset(cfg)
    elif cfg.model_code.lower() == 'ease':
        return EASEDataset(cfg)
    elif cfg.model_code.lower() == 'autoencoder_si':
        return AEDataset(cfg)
    elif cfg.model_code.lower() == 'vae':
        return AEDataset(cfg)
    else:
        raise ValueError('Invalid model name')