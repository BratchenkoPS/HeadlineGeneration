import tensorflow_hub as hub
import logging

from sklearn.manifold import TSNE
from typing import Iterable
from tqdm import tqdm


class Embedder:
    def __init__(self, config: dict):
        logging.info('Staring to load Universal Sentence Encoder and build embeddings')
        self.use = hub.load(config['use_url'])
        self.tsne = TSNE(**config['tsne_config'])

    def get_embeddings(self, list_of_text: Iterable):
        use_embeddings = []
        for text in tqdm(list_of_text):
            use_embeddings.append(self.use([text]).numpy()[0])  # this is just how USE works
        tsne_embeddings = self.tsne.fit_transform(use_embeddings)
        return tsne_embeddings
