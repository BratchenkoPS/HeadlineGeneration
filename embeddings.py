import tensorflow_hub as hub
import logging

from sklearn.manifold import TSNE
from typing import Iterable


class Embedder:
    def __init__(self, config: dict):
        logging.info('Staring to load Universal Sentence Encoder')
        self.use = hub.load(config['use_url'])
        self.tsne = TSNE(**config['tsne_config'])

    def get_embeddings(self, list_of_text: Iterable):
        use_embeddings = []
        for text in list_of_text:
            use_embeddings.append(self.use([text]).numpy()[0])  # this is just how USE works
        tsne_embddings = self.tsne.fit_transform(use_embeddings)
        return tsne_embddings


import yaml
with open('config.yml', 'r') as file:
    config = yaml.load(file)
emb = Embedder(config['Embedder'])
embs = emb.get_embeddings(['huy', 'pizda'])
print(embs)
print(type(embs))
