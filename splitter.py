import pandas as pd

from sklearn.model_selection import train_test_split
from torchtext.data import Field, BucketIterator, TabularDataset
from pathlib import Path
from typing import Tuple


class Splitter:
    """
    Splits the data into train and test, builds fields and iterators
    """

    def __init__(self,
                 result_df: pd.DataFrame,
                 path_to_save_data: str,
                 min_freq: int,
                 test_size: float,
                 batch_size: int,
                 device) -> None:
        """

        Args:
            result_df: dataframe with biggest cluster after clustering, requires to have 'text' and 'title' columns
            path_to_save_data: path to save train/test and combined datasets
            min_freq: minimal frequency of words, if less - word is changed to <unk>
            test_size: test size
            batch_size: batch size
            device: cuda or cpu
        """
        self.result_df = result_df
        self.path_to_save_data = Path(path_to_save_data).absolute()
        self.min_freq = min_freq
        self.test_size = test_size
        self.batch_size = batch_size
        self.device = device

    def get_iterators_and_fields(self) -> Tuple:
        """
        Builds train test/iterators, fields, vocab and tokenizer

        Returns: train/test iterators, data and fields for source/target

        """
        data = self.result_df[['text', 'title']]
        data.columns = ['src', 'trg']
        data.to_csv('data/all_data.csv', index=False)

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
        all_data = TabularDataset(path='data/all_data.csv',
                                  format='csv',
                                  fields=data_fields)

        train, test = train_test_split(data, test_size=self.test_size)
        train.to_csv(self.path_to_save_data.joinpath('train.csv'), index=False)
        test.to_csv(self.path_to_save_data.joinpath('val.csv'), index=False)

        train_data, test_data = TabularDataset.splits(path='data/',
                                                      train='train.csv',
                                                      validation='val.csv',
                                                      format='csv',
                                                      fields=data_fields)
        SRC.build_vocab(all_data, min_freq=self.min_freq)
        TRG.build_vocab(all_data,
                        min_freq=self.min_freq)
        # due to limited amount of data we have to build vocab on all data otherwise we get too much <unk> tokens

        train_iterator, test_iterator = BucketIterator.splits(
            (train_data, test_data),
            batch_size=self.batch_size,
            device=self.device,
            sort=True,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src))

        return train_iterator, test_iterator, train_data, test_data, SRC, TRG

    @staticmethod
    def tokenize(text):
        return [word for word in text.split(' ')]
