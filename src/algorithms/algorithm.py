import abc
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist


class Algorithm:
    __metaclass__ = abc.ABCMeta

    def __init__(self, num_chunks):
        self.num_chunks = num_chunks

    @abc.abstractmethod
    def cluster(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple (centers, clustered_points) with a list of center points of the clusters and a list of points
        sorted into this cluster. "centers" has shape [N, D], where N is the number of points and D is the
        dimensionality of one point. "clustered_points" has dimensionality [N, D].

        :param points: A list of N points with dimensionality D (shape: [N, D])
        :return: A tuple (centers, clustered_points), where centers is a list of cluster center points and
                 clustered_points contains the input points but clustered into the given clusters.
        """
        pass

    @abc.abstractmethod
    def name(self) -> str:
        pass


def cluster_points_by_centers(points: np.ndarray, centers: np.ndarray):
    """
    Return a list of points with shape [N, D] where each point is the clustered version of a point in points.
    :param points: A list of points with shape [N, D]
    :param centers: A list of center points with shape [C, D]
    """
    # dist mat at index (p, c) contains the distance between point[p] and centers[c].
    # Has shape (len(points), len(centers))
    dist_mat = cdist(points, centers, metric='sqeuclidean')

    # Now we have to get the cluster index for each point with the minimal distance
    # cluster_indices is of shape [N,] and contains the index for each point to the closest center point
    cluster_indices = np.argmin(dist_mat, axis=1)

    # now use the indices to map centers to points
    clustered_points = centers[cluster_indices]

    return clustered_points
