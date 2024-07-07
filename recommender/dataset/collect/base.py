import gzip
import pandas as pd
from abc import ABC


class BaseDataReader(ABC):
    """
    Database 내의 특정 시점을 날짜 범위에 따라 가져오는 클래스입니다.
    :input: 
        - data_code (str) - 게임 코드
        - metayn (bool) - 메타 데이터 수집 여부
    """
    def __init__(self, data_code:str, metayn:bool):
        self.data_code = data_code
        self.metayn = metayn

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
    


