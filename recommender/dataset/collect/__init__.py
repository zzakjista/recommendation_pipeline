from .datasource import AmazonGamesReader, SteamGamesReader, MovieLensReader

def data_source_factory(args):
    if args.data_code.lower() == 'amazon_games':
        reader = AmazonGamesReader(args.data_code, args.metayn)
        path = 'recommender/data/Video_Games_5.json.gz' 
        dataset = reader.collect_dataset(path)
    elif args.data_code.lower() == 'steam_games':
        reader = SteamGamesReader(args.data_code, args.metayn)
        path = 'recommender/data/steam-200k.csv'
        dataset = reader.collect_dataset(path)
    elif args.data_code.lower() == 'ml-100k':
        reader = MovieLensReader(args.data_code, args.metayn)
        path = 'recommender/data/ml-100k/u.data'
        dataset = reader.collect_dataset(path)
    else:
        raise ValueError('Invalid game name')
    return dataset
