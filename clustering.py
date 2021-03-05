import pandas as pd
import logging
import json
import plotly.express as px

from sklearn.mixture import BayesianGaussianMixture
from typing import Iterable, List
from scipy.spatial import distance
from tqdm import tqdm
from pathlib import Path


class Clustering:
    def __init__(self,
                 data: pd.DataFrame,
                 directory: str,
                 cluster_picture_name: str,
                 result_data_file_name: str,
                 center_replics_file_name: str,
                 part_to_plot: float,
                 bgm_config: dict):
        self.data = data
        self.directory = Path(directory).absolute()
        self.cluster_picture_name = cluster_picture_name
        self.result_data_file_name = result_data_file_name
        self.center_replics_file_name = center_replics_file_name
        self.part_to_plot = part_to_plot
        self.bgm = BayesianGaussianMixture(**bgm_config)

    def predict(self, vectors: Iterable):
        logging.info('Starting clustering')
        self.bgm.fit(vectors)
        clusters = self.bgm.predict(vectors)
        self.data['x'] = vectors[:, 0]
        self.data['y'] = vectors[:, 1]
        self.data['cluster'] = clusters
        self.data = self.renumber_clusters(self.data)

    def get_center_replics(self):
        center_replics = {}

        for cluster_num in tqdm(self.data['cluster'].value_counts().index):
            cluster_vectors = self.data[self.data['cluster'] == cluster_num][['x', 'y']]
            x_mean = cluster_vectors['x'].mean()
            y_mean = cluster_vectors['y'].mean()
            closest_index = self.get_closest_index([[x_mean, y_mean]], cluster_vectors)
            closest_replic = self.data.iloc[closest_index]['title']
            center_replics[cluster_num] = closest_replic

        self.directory.mkdir(exist_ok=True)
        total_path_to_save = self.directory.joinpath(self.center_replics_file_name)
        with open(total_path_to_save, 'w') as file:
            json.dump(center_replics, file, ensure_ascii=False)

    def plot_clusters(self):
        df_to_plot = self.data.iloc[:0]

        for cluster_num in self.data['cluster'].value_counts().index:
            part_df = self.data[self.data['cluster'] == cluster_num]
            part_df.reset_index(inplace=True, drop=True)
            len_to_allocate = int(len(part_df) * self.part_to_plot)
            indexes_to_allocate = part_df.index[:len_to_allocate]
            part_df = part_df.iloc[indexes_to_allocate]
            df_to_plot = pd.concat([df_to_plot, part_df])

        plot = px.scatter(df_to_plot, x='x', y='y', hover_data=['title'], color='cluster')
        plot.write_html(str(self.directory.joinpath(self.cluster_picture_name)))

    def save_result_df(self):
        biggest_cluster = self.data['cluster'].value_counts().index[0]
        result_df = self.data[self.data['cluster'] == biggest_cluster]
        result_df.to_csv(self.directory.joinpath(self.result_data_file_name))
        return result_df

    def get_clusters_and_final_data(self, vectors):
        self.predict(vectors)
        self.get_center_replics()
        self.plot_clusters()
        result_df = self.save_result_df()
        return result_df

    @staticmethod
    def renumber_clusters(data):
        mapping = {}
        for n, cluster_num in enumerate(data['cluster'].value_counts().index):
            mapping[cluster_num] = n
        data['cluster'] = data['cluster'].apply(lambda x: mapping[x])
        return data

    @staticmethod
    def get_closest_index(node, nodes):
        closest_index = distance.cdist(node, nodes).argmin()
        return closest_index
