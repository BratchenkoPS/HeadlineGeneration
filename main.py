import yaml

from dataloader import DataLoader
from embeddings import Embedder
from clustering import Clustering

if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        config = yaml.load(file)

    loader = DataLoader(config['Dataloader']['url'],
                        config['Dataloader']['path'],
                        config['Dataloader']['name'])

    data = loader.get_data(config['Dataloader']['max_txt_length'],
                           config['Dataloader']['samples'])

    emb = Embedder(config['Embedder'])

    embeddings = emb.get_embeddings(data['title'])

    clustering = Clustering(data,
                            config['Clustering']['directory'],
                            config['Clustering']['cluster_picture_name'],
                            config['Clustering']['result_data_file_name'],
                            config['Clustering']['center_replics_file_name'],
                            config['Clustering']['part_to_plot'],
                            config['Clustering']['bgm_config'])

    df = clustering.get_clusters_and_final_data(embeddings)
