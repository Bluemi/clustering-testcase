import numpy as np
import scipy.cluster.vq
from scipy.cluster.vq import kmeans

from algorithms.algorithm import Algorithm, cluster_points_by_centers


class KMeans(Algorithm):
    def __init__(self, num_chunks):
        super().__init__(num_chunks)

    def cluster(self, points: np.ndarray):
        whitened_points = scipy.cluster.vq.whiten(points)
        centers_whitened = kmeans(whitened_points, self.num_chunks, iter=5)[0]
        centers = centers_whitened * np.std(points, axis=0)  # + np.mean(points, axis=0)
        return centers, cluster_points_by_centers(points, centers)

    def name(self) -> str:
        return 'K-Means'
