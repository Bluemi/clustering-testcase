import time

import numpy as np

from algorithms.algorithm import Algorithm, cluster_points_by_centers


class Popularity(Algorithm):
    def __init__(self, num_chunks, quantization: int = 16):
        super().__init__(num_chunks)
        self.quantization = quantization

    def cluster(self, points):
        quantized_points = (points / self.quantization).astype(int) * self.quantization + self.quantization // 2
        unique_values, counts = np.unique(quantized_points, axis=0, return_counts=True)

        # print(f'unique_values.shape: {unique_values.shape}')
        # print(f'num_chunks: {self.num_chunks}')

        # print(f'counts.shape: {counts.shape}')
        # print(f'unique_values.shape: {unique_values.shape}')
        most_common_indices = np.argsort(counts)[::-1]
        used_most_common_indices = most_common_indices[:self.num_chunks]
        # print(f'used_most_common_indices.shape: {used_most_common_indices.shape}')

        centers = unique_values[used_most_common_indices]
        start_time = time.time()
        clustered_points = cluster_points_by_centers(points, centers)
        end_time = time.time()
        return centers, clustered_points
