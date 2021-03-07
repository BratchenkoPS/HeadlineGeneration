import re
import logging
import json
import wget
import pandas as pd
import gzip

from tqdm import tqdm
from pathlib import Path


class DataLoader:
    """
    Class used to load archive with RIA_news dataset in json format
    """
    def __init__(self, file_url: str, file_directory: str, file_name: str, download: bool) -> None:
        """

        Args:
            file_url: url to file with RIA_news dataset
            file_directory: directory to download and save dataset file
            file_name: dataset file name
            download: whether or not to download file (not needed if you already have it)
        """
        self.file_url = file_url
        self.file_directory = Path(file_directory).absolute()
        self.file_name = file_name
        self.total_file_path = self.file_directory.joinpath(self.file_name)
        self.data = None
        self.download = download

    def download_file(self) -> None:
        """
        Downloads dataset from given URL
        """
        logging.info('Starting to download data')
        self.file_directory.mkdir(exist_ok=True)
        wget.download(self.file_url, str(self.total_file_path))  # wget only works with str sadly

    def load_file(self, max_text_length: int, n_samples: int) -> None:
        """
        Loads json into memory with respect to the given amount of samples and maximum text length

        Args:
            max_text_length: maximum text length filter
            n_samples: the amounts of examples to load

        Returns: None

        """
        self.data = []
        logging.info('Starting to load json into memory')
        with gzip.open(self.total_file_path, 'rt') as file:

            for line in tqdm(file):
                example = json.loads(line)
                example['text'] = self.clean_text(example['text'])
                example['title'] = self.clean_text(example['title'])
                if len(example['text'].split(' ')) < max_text_length:
                    if len(self.data) < n_samples:
                        self.data.append(example)
                    else:
                        break

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans given text with regex, deleting certain artifacts from parsing
        Args:
            text: string with text from dataset

        Returns: fixed string without artifacts

        """
        fixed_text = re.sub('<[^>]+>', ' ', text)  # removing everything inside <>
        fixed_text = re.sub('\n', ' ', fixed_text)
        fixed_text = re.sub('&nbsp;', ' ', fixed_text)
        fixed_text = re.sub('&mdash;', ' ', fixed_text)
        fixed_text = re.sub('\t', ' ', fixed_text)
        fixed_text = re.sub('\r', ' ', fixed_text)
        fixed_text = re.sub('"&hellip;', ' ', fixed_text)
        fixed_text = re.sub('&gt;', ' ', fixed_text)
        fixed_text = re.sub(r'[^\w\s]', ' ', fixed_text)  # removing the punctuation
        fixed_text = re.sub(' +', ' ', fixed_text)  # fix multiple spaces
        return fixed_text

    def get_data(self, max_text_length: int, n_samples: int) -> pd.DataFrame:
        """
        Uses all methods above to download, load and clean dataset with given parameters

        Args:
            max_text_length: maximum text length filter
            n_samples: the amounts of examples to load

        Returns: preprocessed data - dataframe with 3 columns: text, title and length

        """
        if self.download:
            self.download_file()
        self.load_file(max_text_length, n_samples)
        self.data = pd.DataFrame({'text': [sample['text'] for sample in self.data],
                                  'title': [sample['title'] for sample in self.data]})
        self.data.dropna(inplace=True)
        self.data.reset_index(inplace=True, drop=True)

        self.data['length'] = self.data['text'].apply(lambda x: len(x.split(' ')))
        return self.data
