import numpy as np

from algorithms.algorithm import Algorithm, cluster_points_by_centers


class QuadOcTree(Algorithm):
    def __init__(self, num_chunks):
        super().__init__(num_chunks)

    def cluster(self, points: np.ndarray):
        num_dims = points.shape[-1]
        # test_num_dims = 2  # TODO remove

        # test_points = np.random.random((7, test_num_dims)) * 256

        # _split_chunk(test_num_dims, np.array([0] * test_num_dims), np.array([256] * test_num_dims), test_points)
        chunks = [(np.zeros(num_dims), np.zeros(num_dims) + 256, points)]
        while len(chunks) + (2**num_dims) - 1 <= self.num_chunks:
            full_chunk_index, full_chunk = max(enumerate(chunks), key=lambda c: c[1][2].shape[0])
            del chunks[full_chunk_index]  # remove chunk from list
            chunks.extend(
                _split_chunk(num_dims, *full_chunk)
            )

        centers = []
        for chunk in chunks:
            center = (chunk[0] + chunk[1]) / 2
            centers.append(center)

        centers = np.array(centers)

        return centers, cluster_points_by_centers(points, centers)

    def name(self) -> str:
        return 'OcTree'


def _split_chunk(num_dims: int, parent_chunk_start, parent_chunk_end, points):
    """

    :param num_dims: The number of dimensions D for each point.
    :param parent_chunk_start: np.ndarray with shape [D,], marking the beginning of the parent chunk
    :param parent_chunk_end: np.ndarray with shape [D,], marking the end of the parent chunk
    :param points: The points contained in the parent chunk with shape [N, D]
    :return: A list with length 2**D. Each element stands for a child chunk: (chunk_start, chunk_end, points).
    """
    quantization_per_dim = 2
    parent_chunk_size = parent_chunk_end - parent_chunk_start
    child_chunk_size = parent_chunk_size // quantization_per_dim

    # child_chunk_starts has shape [2, D], 2 comes from the two chunks we use for every dimension
    child_chunk_starts = np.linspace(parent_chunk_start, parent_chunk_end, quantization_per_dim, endpoint=False)
    child_chunk_ends = child_chunk_starts + child_chunk_size

    # print(f'before child_chunk_starts:\n{child_chunk_starts}')
    # print(f'before child_chunk_ends:\n{child_chunk_ends}')

    # meshgrid_starts has shape [NC, D], with NC = 2**D (the number of new chunks)
    meshgrid_starts = np.stack(np.meshgrid(*child_chunk_starts.T, indexing='ij'), axis=num_dims).reshape(-1, num_dims)
    meshgrid_ends = np.stack(np.meshgrid(*child_chunk_ends.T, indexing='ij'), axis=num_dims).reshape(-1, num_dims)

    # print(f'points: {points}')
    # print(f'meshgrid_starts:\n{meshgrid_starts}')
    # print(f'meshgrid_ends:\n{meshgrid_ends}')

    # bigger_than_start_indices has shape [N, NC, D]
    bigger_than_start_pos = np.all(meshgrid_starts[None, :] <= points[:, None], axis=2)
    smaller_than_end_pos = np.all(meshgrid_ends[None, :] > points[:, None], axis=2)

    # point contained has shape [N, NC]
    point_contained = np.logical_and(bigger_than_start_pos, smaller_than_end_pos)
    # print(f'bigger_than_start_indices: {bigger_than_start_pos}')
    # print(f'smaller_than_end_pos: {smaller_than_end_pos}')
    # print(f'point_contained:\n{point_contained}')

    chunks = []
    chunk_index = 0
    for start_pos, end_pos in zip(meshgrid_starts, meshgrid_ends):
        chunk_points = points[point_contained[:, chunk_index]]
        chunk = (start_pos, end_pos, chunk_points)
        chunks.append(chunk)
        chunk_index += 1

    return chunks
