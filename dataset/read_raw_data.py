import gzip
import pandas as pd

class DataReader:
    """
    Database 내의 특정 시점을 날짜 범위에 따라 가져오는 클래스입니다.
    """
    def __init__(self, database:str, table:str, date_range:tuple):
        self.database = database
        self.table = table
        self.date_range = date_range

    def read_db(self):
        """
        Database 내의 특정 시점을 날짜 범위에 따라 가져오는 함수입니다.
        """
        pass

    def read_gz_to_pandas(self, path:str):
        """
        gzip 파일을 pandas DataFrame으로 읽어오는 함수입니다.
        """
        gb_file = gzip.open(path,'rb')
        data = pd.read_json(gb_file, lines=True, chunksize=1000)
        data = pd.concat(data)
        return data
    
    def read_csv_to_pandas(self, path:str, **kwargs):
        """
        csv 파일을 pandas DataFrame으로 읽어오는 함수입니다.
        키워드 인자로 sep, header, names 등을 받을 수 있습니다.
        """
        data = pd.read_csv(path, **kwargs)
        return data
    
def data_source_factory(game_name:str):
    reader = DataReader('amazon', 'games_review', ('2012-01-01', '2012-12-31'))
    if game_name.lower() == 'amazon_games':
        data = reader.read_gz_to_pandas('data/Video_Games_5.json.gz')
    elif game_name.lower() == 'steam_games':
        data = reader.read_csv_to_pandas('data/steam-200k.csv', header=None, names=['user_id', 'item_id', 'action', 'rating', 'dummy'])
    else:
        raise ValueError('Invalid game name')
    return data

