import tensorflow_hub as hub
import logging
import wget
import tarfile

from sklearn.manifold import TSNE
from typing import Iterable
from tqdm import tqdm
from pathlib import Path


class Embedder:
    def __init__(self, config: dict):
        logging.info('Staring to load Universal Sentence Encoder and build embeddings')
        self.use_url = config['use_url']
        self.use_directory = Path(config['use_directory']).absolute()
        self.use_model = None
        self.tsne = TSNE(**config['tsne_config'])

    def get_embeddings(self, list_of_text: Iterable):
        self.download_use()
        self.load_use_model()
        use_embeddings = []
        for text in tqdm(list_of_text):
            use_embeddings.append(self.use_model([text]).numpy()[0])  # this is just how USE works
        tsne_embeddings = self.tsne.fit_transform(use_embeddings)
        return tsne_embeddings

    def download_use(self):
        self.use_directory.mkdir(exist_ok=True)
        wget.download(self.use_url, str(self.use_directory.joinpath('use_model_v4.tar.gz')))

        tar = tarfile.open(self.use_directory.joinpath('use_model_v4.tar.gz'), "r:gz")
        tar.extractall(str(self.use_directory))

        tar.close()

    def load_use_model(self):
        self.use_model = hub.load(str(self.use_directory))


