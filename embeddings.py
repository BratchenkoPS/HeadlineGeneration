import tensorflow_hub as hub
import logging
import wget
import tarfile

from sklearn.manifold import TSNE
from typing import List
from tqdm import tqdm
from pathlib import Path


class Embedder:
    """
    Class used to download and load google universal sentence encoder, build embeddings for text and preprocess them
     with TSNE for clustering
    """

    def __init__(self, config: dict) -> None:
        """

        Args:
            config: dict with parameters: url, directory, filename, download flag and TSNE config
        """
        logging.info('Staring to load Universal Sentence Encoder and build embeddings')
        self.use_url = config['use_url']
        self.use_directory = Path(config['use_directory']).absolute()
        self.use_file_name = config['use_file_name']
        self.use_model = None
        self.tsne = TSNE(**config['tsne_config'])
        self.download = config['download']

    def get_embeddings(self, list_of_text: List) -> List:
        """
        Builds embeddings from given list of text, first - USE embeddings, then - TSNE.
        Args:
            list_of_text: given list of text for vectorizing

        Returns: List of 2d vectors for each sentence.

        """
        if self.download:
            self.download_use()
        self.use_model = hub.load(str(self.use_directory))
        use_embeddings = []
        for text in tqdm(list_of_text):
            use_embeddings.append(self.use_model([text]).numpy()[0])  # this is just how USE works
        tsne_embeddings = self.tsne.fit_transform(use_embeddings)
        return tsne_embeddings

    def download_use(self) -> None:
        """
        Downloads and extract google USE v4.
        Returns: None

        """
        self.use_directory.mkdir(parents=True, exist_ok=True)
        wget.download(self.use_url, str(self.use_directory.joinpath(self.use_file_name)))

        tar = tarfile.open(self.use_directory.joinpath(self.use_file_name), "r:gz")
        tar.extractall(str(self.use_directory))

        tar.close()
