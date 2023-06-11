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

        most_common_indices = np.argsort(counts)[::-1]
        used_most_common_indices = most_common_indices[:self.num_chunks]

        centers = unique_values[used_most_common_indices]
        clustered_points = cluster_points_by_centers(points, centers)

        return centers, clustered_points
