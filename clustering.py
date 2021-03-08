import pandas as pd
import logging
import json
import plotly.express as px

from sklearn.mixture import BayesianGaussianMixture
from typing import List
from scipy.spatial import distance
from tqdm import tqdm
from pathlib import Path


class Clustering:
    """
    Class used to clusterize given list of text with TSNE embeddings
    """

    def __init__(self,
                 data: pd.DataFrame,
                 directory: str,
                 cluster_picture_name: str,
                 result_data_file_name: str,
                 center_replics_file_name: str,
                 part_to_plot: float,
                 bgm_config: dict) -> None:
        """

        Args:
            data: dataframe with text, title and length
            directory: folder to save clusters and result frame
            cluster_picture_name: name of html file with clustering picture
            result_data_file_name: name of result dataframe with biggest cluster
            center_replics_file_name: name of json file with center replic for each cluster
            part_to_plot: part of data to plot (not recommended to plot more than 10k-ish replics)
            bgm_config: config for gaussian mixtures
        """
        self.data = data
        self.directory = Path(directory).absolute()
        self.cluster_picture_name = cluster_picture_name
        self.result_data_file_name = result_data_file_name
        self.center_replics_file_name = center_replics_file_name
        self.part_to_plot = part_to_plot
        self.bgm = BayesianGaussianMixture(**bgm_config)

    def predict(self, vectors: List) -> None:  # TODO rework to work with self.data to avoid order problems
        """
        Trains and predicts clusters for given list of vectors (must be same order as in self.data)
        Args:
            vectors: list of TSNE embeddings

        Returns: None

        """
        logging.info('Starting clustering')
        self.bgm.fit(vectors)
        clusters = self.bgm.predict(vectors)
        self.data['x'] = vectors[:, 0]
        self.data['y'] = vectors[:, 1]
        self.data['cluster'] = clusters
        self.data = self.renumber_clusters(self.data)

    def get_center_replics(self) -> None:
        """
        Finds the center texts for each cluster and saves it into json file like so:
        0: central text for cluster 0
        1: central text for cluster 1

        Returns: None

        """
        center_replics = {}

        for cluster_num in tqdm(self.data['cluster'].value_counts().index):
            cluster_vectors = self.data[self.data['cluster'] == cluster_num][['x', 'y']]
            x_mean = cluster_vectors['x'].mean()
            y_mean = cluster_vectors['y'].mean()
            closest_index = self.get_closest_index([[x_mean, y_mean]], cluster_vectors)
            closest_replic = self.data.iloc[closest_index]['title']
            center_replics[cluster_num] = closest_replic
            logging.info('Central replic for cluster number {} is: {}'.format(cluster_num, closest_replic))

        self.directory.mkdir(parents=True, exist_ok=True)
        total_path_to_save = self.directory.joinpath(self.center_replics_file_name)

        with open(total_path_to_save, 'w') as file:
            json.dump(center_replics, file, ensure_ascii=False)

    def plot_clusters(self) -> None:
        """
        Plots a given data part of each cluster into html file for later analysis

        Returns: None

        """
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

    def save_result_df(self) -> pd.DataFrame:
        """
        Saves the biggest cluster data into dataframe

        Returns: dataframe with data for biggest cluster

        """
        biggest_cluster = self.data['cluster'].value_counts().index[0]
        result_df = self.data[self.data['cluster'] == biggest_cluster]
        result_df.to_csv(self.directory.joinpath(self.result_data_file_name))
        return result_df

    def get_clusters_and_final_data(self, vectors: List) -> pd.DataFrame:
        """
        Method that combines all the the methods above to plot clusters, find central replics and save the biggest
         cluster for further work
        Args:
            vectors: list of TSNE embeddings for clusterization

        Returns:

        """
        self.predict(vectors)
        self.get_center_replics()
        self.plot_clusters()
        result_df = self.save_result_df()
        return result_df

    @staticmethod
    def renumber_clusters(data: pd.DataFrame) -> pd.DataFrame:
        """
        Renumbers the cluster. The gaussian mixtures might cut some of the clusters without renumbering it, so you end
         up with cluster numbers like 1, 3, 5 instead 0, 1, 2. This methods fixes it.
        Args:
            data: dataframe with text and clusters

        Returns: dataframe with fixed cluster nums

        """
        mapping = {}
        for n, cluster_num in enumerate(data['cluster'].value_counts().index):
            mapping[cluster_num] = n
        data['cluster'] = data['cluster'].apply(lambda x: mapping[x])
        return data

    @staticmethod
    def get_closest_index(node: List, nodes: List[List]) -> int: # TODO make proper typization
        """
        Gets the index of the closest replic to the certain cluster center
        Args:
            node: coordinates of the cluster center
            nodes: list of coordinates of all replic in cluster

        Returns: index of the closest replic

        """
        closest_index = distance.cdist(node, nodes).argmin()
        return closest_index
