import abc
from typing import Optional

import numpy as np


def normalize_to_color_space(points):
    points = points - np.min(points)
    points = points / np.max(points)
    return points * 255


class Distribution:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate_points(self) -> np.ndarray:
        """
        Generates points with a minimum of 0 and a maximum of 255.
        :return:
        """
        pass


class NormalDistribution(Distribution):
    def __init__(self, num_points: int, num_dims: int, seed: Optional[int] = 42):
        self.num_points = num_points
        self.num_dims = num_dims
        self.seed = seed

    def generate_points(self) -> np.ndarray:
        if self.seed:
            np.random.seed(self.seed)
        points = np.random.normal(size=(self.num_points, self.num_dims))
        return normalize_to_color_space(points)


class UniformDistribution(Distribution):
    def __init__(self, num_points: int, num_dims: int, seed: Optional[int] = 42):
        self.num_points = num_points
        self.num_dims = num_dims
        self.seed = seed

    def generate_points(self) -> np.ndarray:
        if self.seed:
            np.random.seed(self.seed)
        points = np.random.random(size=(self.num_points, self.num_dims))
        return normalize_to_color_space(points)


class CenteredDistribution(Distribution):
    def __init__(self, num_centers: int, num_points: int, num_dims: int, seed: Optional[int] = 42):
        self.num_centers = num_centers
        self.num_points = num_points
        self.num_dims = num_dims
        self.seed = seed

    def generate_points(self) -> np.ndarray:
        points_per_center = int(np.ceil(self.num_points / self.num_centers))
        if self.seed:
            np.random.seed(self.seed)

        centers = np.random.normal(size=(self.num_centers, self.num_dims))
        dists = np.abs(np.random.normal(size=self.num_centers)) * 0.3

        point_buffer = []
        for center, dist in zip(centers, dists):
            points = np.random.normal(loc=center, scale=dist, size=(points_per_center, self.num_dims))
            point_buffer.append(points)

        points = np.concatenate(point_buffer)[:self.num_points]

        return normalize_to_color_space(points)
