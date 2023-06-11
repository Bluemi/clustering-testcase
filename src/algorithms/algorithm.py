import abc
from typing import Tuple

import numpy as np


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
