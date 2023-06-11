import abc
from typing import Optional

import numpy as np


def normalize_to_color_space(points):
    points = points - np.min(points)
    points = points / np.max(points)
    return points * 255


class Distribution:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_points(self) -> np.ndarray:
        """
        Generates points with a minimum of 0 and a maximum of 255.
        :return:
        """
        pass


class GaussDistribution(Distribution):
    def __init__(self, num_points: int, num_dims: int, seed: Optional[int] = 42):
        super().__init__()
        self.num_points = num_points
        self.num_dims = num_dims
        self.seed = seed

    def generate_points(self) -> np.ndarray:
        if self.seed:
            np.random.seed(self.seed)
        points = np.random.normal(size=(self.num_points, self.num_dims))
        return normalize_to_color_space(points)
