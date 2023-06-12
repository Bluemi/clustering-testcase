# clustering-testcase

Implementation of some clustering algorithms.

## Controls
- `c`: Toggles visualization of clustering algorithm
- `+` / `-`: Use more or less cluster centroids
- `a` / `A`: Use next/prev cluster algorithm
- `n` / `N`: Use next/prev image/distribution
- `2`: Show 2d point distribution
- `3`: Show images. Colors are interpreted as 3d points and clustered
- `r`: resample 2d distribution
- `Esc`: Quit the program

## Credits
- The generalized Lloyd Algorithm is taken from [spencerkent](https://github.com/spencerkent/generalized-lloyd-quantization/tree/6d27b1b1a16a128104f224b06ee8361f6ed600d9), but I did not manage to get it running.
- The K-Means Clustering uses the [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html#scipy.cluster.vq.kmeans) implementation.