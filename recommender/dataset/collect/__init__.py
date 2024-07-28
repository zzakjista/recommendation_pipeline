from .datasource import AmazonGamesReader, SteamGamesReader, MovieLensReader

def data_source_factory(cfg):
    if cfg.data.data_code.lower() == 'amazon_games':
        reader = AmazonGamesReader(cfg.data.data_code, cfg.data.metayn)
        path = 'recommender/data/Video_Games_5.json.gz' 
        dataset = reader.collect_dataset(path)
    elif cfg.data.data_code.lower() == 'steam_games':
        reader = SteamGamesReader(cfg.data.data_code, cfg.data.metayn)
        path = 'recommender/data/steam-200k.csv'
        dataset = reader.collect_dataset(path)
    elif cfg.data.data_code.lower() == 'ml-100k':
        reader = MovieLensReader(cfg.data.data_code, cfg.data.metayn)
        path = 'recommender/data/ml-100k/u.data'
        dataset = reader.collect_dataset(path)
    else:
        raise ValueError('Invalid game name')
    return dataset
