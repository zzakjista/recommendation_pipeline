from .base import BaseDataReader

class AmazonGamesReader(BaseDataReader):
    def __init__(self, data_code:str, metayn:bool):
        super().__init__(data_code, metayn)
    
    def read_gz_to_pandas(self, path:str):
        return super().read_gz_to_pandas(path)
    
    def read_csv_to_pandas(self, path:str, **kwargs):
        return super().read_csv_to_pandas(path, **kwargs)
    
    def collect_dataset(self, path:str):
        dataset = {} 
        interaction = self.read_interaction_data(path)
        dataset['interaction'] = interaction
        return dataset
    
    def read_interaction_data(self, path:str):
        return self.read_gz_to_pandas(path)
    
    def read_meta_data(self, path:str):
        raise NotImplementedError('Not ready yet')
    
class SteamGamesReader(BaseDataReader):
    def __init__(self, data_code:str, metayn:bool):
        super().__init__(data_code, metayn)
    
    def read_gz_to_pandas(self, path:str):
        return super().read_gz_to_pandas(path)
    
    def read_csv_to_pandas(self, path:str, **kwargs):
        return super().read_csv_to_pandas(path, **kwargs)
    
    def collect_dataset(self, path:str):
        dataset = {} 
        interaction = self.read_interaction_data(path)
        dataset['interaction'] = interaction
        return dataset
    
    def read_interaction_data(self, path:str):
        # path : 'data/steam-200k.csv'
        return self.read_csv_to_pandas(path, header=None, names=['user_id', 'item_id', 'action', 'rating', 'dummy'])
    
    def read_meta_data(self, path:str):
        raise NotImplementedError('Not ready yet')
    

class MovieLensReader(BaseDataReader):
    def __init__(self, data_code:str, metayn:bool):
        super().__init__(data_code, metayn)
    
    def read_gz_to_pandas(self, path:str):
        return super().read_gz_to_pandas(path)
    
    def read_csv_to_pandas(self, path:str, **kwargs):
        return super().read_csv_to_pandas(path, **kwargs)
    
    def collect_dataset(self, path:str):
        dataset = {}
        interaction = self.read_interaction_data(path)
        dataset['interaction'] = interaction
        if self.metayn:
            meta_dataset = self.read_meta_data()
            dataset['meta'] = meta_dataset
        return dataset
    
    def read_interaction_data(self, path:str):
        return self.read_csv_to_pandas('recommender/data/ml-100k/u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    def read_meta_data(self):
        meta_dataset = {}
        item = self.read_csv_to_pandas('recommender/data/ml-100k/u.item', sep='|', header=None, encoding='ISO-8859-1', names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        user = self.read_csv_to_pandas('recommender/data/ml-100k/u.user', sep='|', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip_code'] )
        genre = self.read_csv_to_pandas('recommender/data/ml-100k/u.genre', sep='|', header=None, names=['genre', 'id'])
        occupation = self.read_csv_to_pandas('recommender/data/ml-100k/u.occupation', sep='|', header=None, names=['occupation'])
        meta_dataset['item'] = item
        meta_dataset['user'] = user
        meta_dataset['genre'] = genre
        meta_dataset['occupation'] = occupation
        return meta_dataset
