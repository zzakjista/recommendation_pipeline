import pandas as pd 
from abc import ABC, abstractmethod
import pickle
# Abstract class for data processing
class BasePreProcessor(ABC):
    """
    Abstract class for data processing
    모든 preprocessor는 이 클래스를 상속받아야 한다.
    """
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.processed_data = None

    @abstractmethod
    def preprocess(self):
        pass
    
    @abstractmethod
    def save(self):
        pass


class AmazonGamesPreprocessor(BasePreProcessor):
    """
    AmazonGames 데이터 전처리 클래스
    input : raw_data : pd.DataFrame
    output : interaction : pd.DataFrame
    """
    def __init__(self, dataset, cfg):
        super().__init__(dataset, cfg)
        self.cfg = cfg 


    def preprocess(self):
        interaction = self._ppc_interaction()
        self.dataset['interaction'] = interaction
        return self.dataset

    def _ppc_meta(self):
        pass

    def _ppc_interaction(self):
        self.processed_data = self.dataset['interaction'][['reviewerID', 'asin', 'overall', 'reviewTime']]
        self.processed_data = (self.processed_data
                       .rename(columns={'reviewerID': 'user_id', 
                                    'asin': 'item_id', 'overall': 'rating', 
                                    'reviewTime': 'timestamp'}))
        self.processed_data = self.processed_data.sort_values(by='timestamp')
        self.processed_data = self._implicit_feedback()
        self.processed_data = self.processed_data.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
        self.processed_data = self.processed_data.reset_index(drop=True)
        return self.processed_data

    def _implicit_feedback(self):
        if 'rating' not in self.processed_data.columns:
            raise ValueError('Rating column does not exist')
        self.processed_data['rating'] = self.processed_data['rating'].apply(lambda x: 1 if x >= 4 else 0)
        return self.processed_data
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.dataset, f)
    
class SteamGamesPreprocessor(BasePreProcessor):
    """
    SteamGames 데이터 전처리 클래스
    :input: 
        - dataset
    :output: 
        - dataset (preprocessed)
    """
    def __init__(self, dataset, cfg):
        super().__init__(dataset, cfg)
        self.cfg = cfg 

    def preprocess(self):
        interaction = self._ppc_interaction()
        self.dataset['interaction'] = interaction
        return self.dataset
    
    def _ppc_meta(self):
        pass
    
    def _ppc_interaction(self):
        self.processed_data = self.dataset['interaction']
        self.processed_data = self.processed_data.rename(columns={'user_id': 'user_id', 'item_id': 'item_id', 'action': 'action', 'rating': 'rating', 'dummy': 'dummy'})
        self.processed_data = self.processed_data.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
        self.processed_data = self.processed_data.reset_index(drop=True)
        return self.processed_data

    def _implicit_feedback(self):
        if 'rating' not in self.processed_data.columns:
            raise ValueError('Rating column does not exist')
        self.processed_data['rating'] = self.processed_data['rating'].apply(lambda x: 1 if x >= 4 else 0)
        return self.processed_data
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.dataset, f)
    
class MovieLensPreprocessor(BasePreProcessor):
    """
    MovieLens 데이터 전처리 클래스
    :input: 
        - dataset
    :output: 
        - dataset (preprocessed)
    """
    def __init__(self, dataset, cfg):
        super().__init__(dataset, cfg)
        self.cfg = cfg 

    def preprocess(self):
        interaction = self._ppc_interaction()
        self.dataset['interaction'] = interaction
        if self.cfg.metayn:
            meta = self._ppc_meta()
            self.dataset['meta'] = meta
        return self.dataset
    
    def _ppc_meta(self):
        # item 장르를 nested하게 변경
        item = self.dataset['meta']['item']
        item = item.set_index('item_id')
        item = item.drop(['title', 'release_date', 'video_release_date', 'IMDb_URL'], axis=1)
        item = item.stack().reset_index()
        item.columns = ['item_id', 'genre', 'value']
        item = item[item['value'] == 1].drop('value', axis=1)
        genre = self.dataset['meta']['genre']
        genre = genre.replace("Children's", "Children")
        genre2idx = {genre: idx for idx, genre in enumerate(genre['genre'])}
        item['genre_idx'] = item['genre'].map(genre2idx)
        item['genre_indices'] = item.groupby('item_id')['genre_idx'].apply(list)
        item = item.groupby('item_id')['genre_idx'].apply(list).rename('genre_indices').reset_index()
        user = self.dataset['meta']['user']
        user = user[['user_id', 'age']]
        user['age'] = (user['age'] - user['age'].mean()) / user['age'].std()
        self.dataset['meta'] = {'item': item, 'user': user}
        return self.dataset['meta']
    
    def _ppc_interaction(self):
        interaction = self.dataset['interaction']
        interaction = interaction.rename(columns={'user_id': 'user_id', 'item_id': 'item_id', 'action': 'action', 'rating': 'rating', 'dummy': 'dummy'})
        interaction = interaction.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
        interaction = interaction.reset_index(drop=True)
        interaction['rating'] = interaction['rating'].apply(lambda x : 1 if x > 3 else 0)
        interaction = interaction[interaction['rating'] == 1]
        return interaction

    def _implicit_feedback(self):
        if 'rating' not in self.processed_data.columns:
            raise ValueError('Rating column does not exist')
        self.processed_data['rating'] = self.processed_data['rating'].apply(lambda x: 1 if x >= 4 else 0)
        return self.processed_data
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.dataset, f)