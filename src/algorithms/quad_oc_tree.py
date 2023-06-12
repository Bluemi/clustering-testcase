import numpy as np

from algorithms.algorithm import Algorithm


class QuadOcTree(Algorithm):
    def __init__(self, num_chunks):
        super().__init__(num_chunks)

    def cluster(self, points: np.ndarray):
        num_dims = points.shape[-1]
        point_indices = np.arange(len(points))
        chunks = [(np.zeros(num_dims), np.zeros(num_dims) + 256, point_indices)]
        while len(chunks) + (2**num_dims) - 1 <= self.num_chunks:
            full_chunk_index, full_chunk = max(enumerate(chunks), key=lambda c: c[1][2].shape[0])
            del chunks[full_chunk_index]  # remove chunk from list
            chunks.extend(
                _split_chunk(num_dims, *full_chunk, points)
            )

        centers = []
        clustered_points = np.empty_like(points)
        for chunk in chunks:
            center = (chunk[0] + chunk[1]) / 2
            centers.append(center)
            clustered_points[chunk[2]] = center

        centers = np.array(centers)
        return centers, clustered_points

    def name(self) -> str:
        return 'OcTree'


def _split_chunk(num_dims: int, parent_chunk_start, parent_chunk_end, point_indices, points):
    """

    :param num_dims: The number of dimensions D for each point.
    :param parent_chunk_start: np.ndarray with shape [D,], marking the beginning of the parent chunk
    :param parent_chunk_end: np.ndarray with shape [D,], marking the end of the parent chunk
    :param point_indices: The indices into points of the given parent chunk with shape [N, D]
    :param points: All points. To get points of parent chunk index with point_indices
    :return: A list with length 2**D. Each element stands for a child chunk: (chunk_start, chunk_end, points).
    """
    quantization_per_dim = 2
    parent_chunk_size = parent_chunk_end - parent_chunk_start
    child_chunk_size = parent_chunk_size // quantization_per_dim

    # child_chunk_starts has shape [2, D], 2 comes from the two chunks we use for every dimension
    child_chunk_starts = np.linspace(parent_chunk_start, parent_chunk_end, quantization_per_dim, endpoint=False)
    child_chunk_ends = child_chunk_starts + child_chunk_size

    # meshgrid_starts has shape [NC, D], with NC = 2**D (the number of new chunks)
    meshgrid_starts = np.stack(np.meshgrid(*child_chunk_starts.T, indexing='ij'), axis=num_dims).reshape(-1, num_dims)
    meshgrid_ends = np.stack(np.meshgrid(*child_chunk_ends.T, indexing='ij'), axis=num_dims).reshape(-1, num_dims)

    # bigger_than_start_indices has shape [N, NC, D]
    bigger_than_start_pos = np.all(meshgrid_starts[None, :] <= points[point_indices, None], axis=2)
    smaller_than_end_pos = np.all(meshgrid_ends[None, :] > points[point_indices, None], axis=2)

    # point contained has shape [N, NC]
    point_contained = np.logical_and(bigger_than_start_pos, smaller_than_end_pos)

    chunks = []
    chunk_index = 0
    for start_pos, end_pos in zip(meshgrid_starts, meshgrid_ends):
        chunk_point_indices = point_indices[point_contained[:, chunk_index]]
        chunk = (start_pos, end_pos, chunk_point_indices)
        chunks.append(chunk)
        chunk_index += 1

    return chunks
