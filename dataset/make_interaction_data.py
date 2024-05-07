import pandas 
from abc import ABC, abstractmethod

# Abstract class for data processing
class BasePreProcessor(ABC):
    """
    Abstract class for data processing
    모든 preprocessor는 이 클래스를 상속받아야 한다.
    """
    def __init__(self, data):
        self.data = data
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
    def __init__(self, data):
        super().__init__(data)


    def preprocess(self):
        self.processed_data = self.data[['reviewerID', 'asin', 'overall', 'reviewTime']]
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
        self.processed_data.to_pickle(path)
        print(f'Saved to {path}')
    
class SteamGamesPreprocessor(BasePreProcessor):
    """
    AmazonGames 데이터 전처리 클래스
    input : raw_data : pd.DataFrame
    output : interaction : pd.DataFrame
    """
    def __init__(self, data):
        super().__init__(data)


    def preprocess(self):
        self.processed_data = self.data.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
        self.processed_data['rating'] = 1
        self.processed_data = self.processed_data.reset_index(drop=True)
        return self.processed_data

    def _implicit_feedback(self):
        if 'rating' not in self.processed_data.columns:
            raise ValueError('Rating column does not exist')
        self.processed_data['rating'] = self.processed_data['rating'].apply(lambda x: 1 if x >= 4 else 0)
        return self.processed_data
    
    def save(self, path):
        self.processed_data.to_pickle(path)
        print(f'Saved to {path}')
    
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