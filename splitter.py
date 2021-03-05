import pandas as pd

from sklearn.model_selection import train_test_split
from torchtext.data import Field, BucketIterator, TabularDataset
from pathlib import Path


class Splitter:
    def __init__(self,
                 result_df: pd.DataFrame,
                 path_to_save_data: str,
                 min_freq: int,
                 test_size: float,
                 batch_size: int,
                 device):
        self.result_df = result_df
        self.path_to_save_data = Path(path_to_save_data).absolute()
        self.min_freq = min_freq
        self.test_size = test_size
        self.batch_size = batch_size
        self.device = device

    def get_iterators_and_fields(self):
        data = self.result_df[['text', 'title']]
        data.columns = ['src', 'trg']

        SRC = Field(tokenize=self.tokenize,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)

        TRG = Field(tokenize=self.tokenize,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)

        data_fields = [('src', SRC), ('trg', TRG)]

        train, test = train_test_split(data, test_size=self.test_size)
        train.to_csv(self.path_to_save_data.joinpath('train.csv'), index=False)
        test.to_csv(self.path_to_save_data.joinpath('val.csv'), index=False)

        train_data, test_data = TabularDataset.splits(path='data/',
                                                      train='train.csv',
                                                      validation='val.csv',
                                                      format='csv',
                                                      fields=data_fields)
        SRC.build_vocab(train_data, min_freq=self.min_freq)
        TRG.build_vocab(test_data, min_freq=self.min_freq)

        train_iterator, test_iterator = BucketIterator.splits(
            (train_data, test_data),
            batch_size=self.batch_size,
            device=self.device)

        return train_iterator, test_iterator, train_data, test_data, SRC, TRG

    @staticmethod
    def tokenize(text):
        return [word for word in text.split(' ')]
