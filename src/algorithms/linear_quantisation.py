import numpy as np

from algorithms.algorithm import Algorithm


class LinearQuantization(Algorithm):
    def __init__(self, num_chunks):
        super().__init__(num_chunks)

    def cluster(self, points):
        num_dims = points.shape[-1]
        qpd_float = np.power(self.num_chunks, 1/num_dims)
        quantization_per_dim = int(qpd_float)
        if abs(qpd_float - quantization_per_dim - 1) < 0.001:
            quantization_per_dim += 1

        chunk_size = 256 / quantization_per_dim
        chunk_starts = np.linspace(0, 256, quantization_per_dim, endpoint=False)
        chunk_centers = chunk_starts + chunk_size / 2

        chunk_centers = np.stack(np.meshgrid(*([chunk_centers] * num_dims), indexing='ij'), axis=num_dims)

        centered_points = np.floor(points / chunk_size) * chunk_size + chunk_size / 2

        return chunk_centers.reshape(-1, num_dims), centered_points

    def name(self) -> str:
        return 'Linear Quantization'
