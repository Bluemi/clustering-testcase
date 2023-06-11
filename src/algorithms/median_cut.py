from typing import Tuple, List

import numpy as np

from algorithms.algorithm import Algorithm


class Chunk:
    def __init__(self, point_indices: np.ndarray):
        self.point_indices = point_indices
        self.can_split = True

    def get_wrapper_max(self, points: np.ndarray):
        """
        start, end = get_wrapper(points[self.point_indices])
        size = end - start
        max_index = np.argmax(size)
        return size[max_index], max_index
        """
        my_points = points[self.point_indices]
        std = np.std(my_points, axis=0)
        max_index = np.argmax(std)
        return std[max_index], max_index

    def get_median(self, axis, points):
        median = np.median(points[self.point_indices, axis])
        if median - np.min(points[self.point_indices, axis]) < 0.0001:
            median += 0.0001
        elif np.max(points[self.point_indices, axis]) - median < 0.0001:
            median -= 0.0001
        return median

    def get_mean(self, points: np.ndarray):
        return np.mean(points[self.point_indices], axis=0)

    def split_point_indices(self, split_value, split_axis, points):
        lower_indices = points[self.point_indices, split_axis] < split_value
        higher_indices = np.logical_not(lower_indices)
        return self.point_indices[lower_indices], self.point_indices[higher_indices]


class MedianCut(Algorithm):
    def __init__(self, num_chunks):
        super().__init__(num_chunks)

    def cluster(self, points: np.ndarray):
        chunks = [Chunk(np.arange(len(points)))]
        while len(chunks) + 1 <= self.num_chunks:
            wrapper = [c.get_wrapper_max(points) for c in chunks if c.can_split]
            max_chunk_index, max_wrapper = max(enumerate(wrapper), key=lambda w: w[1][0])
            if max_wrapper[0] < 0.001:
                break
            max_chunk = chunks[max_chunk_index]
            del chunks[max_chunk_index]
            chunk_result = _split_chunk(max_chunk, max_wrapper[1], points)
            chunks.extend(chunk_result)

        result_points = np.empty_like(points)

        centers = []
        for chunk in chunks:
            cluster_center = chunk.get_mean(points)
            result_points[chunk.point_indices] = cluster_center
            centers.append(cluster_center)

        centers = np.array(centers)

        return centers, result_points

    def name(self) -> str:
        return 'Median Cut'


def get_wrapper(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.min(points, axis=0), np.max(points, axis=0)


def _split_chunk(chunk: Chunk, split_axis, points) -> List[Chunk]:
    median = chunk.get_median(split_axis, points)
    point_indices1, point_indices2 = chunk.split_point_indices(median, split_axis, points)
    chunks = []
    one_invalid = False
    if point_indices1.shape[0] > 0:
        chunks.append(Chunk(point_indices1))
    else:
        one_invalid = True
    if point_indices2.shape[0] > 0:
        chunks.append(Chunk(point_indices2))
    else:
        one_invalid = True

    if one_invalid:
        chunks[0].can_split = False
    return chunks

