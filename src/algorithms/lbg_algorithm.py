"""
The implementation of this algorithm is used from https://github.com/internaut/py-lbg/blob/master/lbg.py
"""
import numpy as np
import scipy.cluster.vq
import algorithms.lbg

from algorithms.algorithm import Algorithm, cluster_points_by_centers


class LBGAlg(Algorithm):
    def __init__(self, num_chunks):
        super().__init__(num_chunks)

    def cluster(self, points: np.ndarray):
        whitened_points = scipy.cluster.vq.whiten(points)
        using_num_chunks = int(2**np.floor(np.log2(self.num_chunks)))
        centers_whitened, _, _ = algorithms.lbg.generate_codebook(
            whitened_points, using_num_chunks, epsilon=0.01
        )

        centers = centers_whitened * np.std(points, axis=0)

        return centers, cluster_points_by_centers(points, centers)

    def name(self) -> str:
        return 'Linde-Buzo-Gray'
