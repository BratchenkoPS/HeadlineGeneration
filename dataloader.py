import re
import logging
import json
import wget
import pandas as pd

from tqdm import tqdm
from pathlib import Path


class DataLoader:
    def __init__(self, file_url: str, file_directory: str, file_name: str):
        self.file_url = file_url
        self.file_directory = Path(file_directory).absolute()
        self.file_name = file_name
        self.total_file_path = self.file_directory.joinpath(self.file_name)
        self.data = None

    def download_file(self):
        logging.info('Starting to download data')
        self.file_directory.mkdir(exist_ok=True)
        wget.download(self.file_url, str(self.total_file_path))  # wget only works with str sadly

    def load_file(self, max_text_length: int, n_samples: int):
        self.data = []
        logging.info('Starting to load json into memory')

        with open(self.total_file_path, 'r') as file:

            for line in tqdm(file):
                example = json.loads(line)

                if len(self.data) < n_samples:
                    length = len(example['text'].split())

                    if length <= max_text_length:
                        example['length'] = length
                        self.data.append(example)
                else:
                    break

    @staticmethod
    def clean_text(text: str):
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

    def clean_data(self):
        logging.info('Starting data cleaning')
        for n, sample in tqdm(enumerate(self.data)):
            text = sample['text']
            title = sample['title']
            self.data[n]['text'] = self.clean_text(text)
            self.data[n]['title'] = self.clean_text(title)

    def get_data(self, max_text_length: int, n_samples: int):
        self.download_file()
        self.load_file(max_text_length, n_samples)
        self.clean_data()
        self.data = pd.DataFrame({'text': [sample['text'] for sample in self.data],
                                  'title': [sample['title'] for sample in self.data],
                                  'length': [sample['length'] for sample in self.data]})
        return self.data
