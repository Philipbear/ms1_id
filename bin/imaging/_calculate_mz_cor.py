"""
The first run of this script might be slower due to Numba's compilation time,
but subsequent runs should be much faster.
"""

import os
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, save_npz


@njit
def mz_correlation_numba(intensities1, intensities2, min_spec_overlap_ratio=0.6):
    x_non_zero_mask = intensities1 != 0
    y_non_zero_mask = intensities2 != 0

    non_zero_mask = (intensities1 != 0) & (intensities2 != 0)

    n = len(intensities1[non_zero_mask])
    smaller_length = min(len(intensities1[x_non_zero_mask]), len(intensities2[y_non_zero_mask]))
    if n < min_spec_overlap_ratio * smaller_length:
        return 0.0

    x = intensities1[non_zero_mask]
    y = intensities2[non_zero_mask]

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    sum_y2 = np.sum(y * y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

    return numerator / denominator if denominator != 0 else 1.0


@njit
def calc_mz_correlation_matrix_numba(intensity_matrix, min_spec_overlap_ratio=0.6):
    n_mzs = intensity_matrix.shape[0]
    rows = np.zeros(n_mzs * n_mzs, dtype=np.int32)
    cols = np.zeros(n_mzs * n_mzs, dtype=np.int32)
    data = np.zeros(n_mzs * n_mzs, dtype=np.float64)
    counter = 0

    for i in range(n_mzs):
        for j in range(i + 1, n_mzs):
            corr = mz_correlation_numba(intensity_matrix[i], intensity_matrix[j], min_spec_overlap_ratio)
            rows[counter] = i
            cols[counter] = j
            data[counter] = corr
            counter += 1

    return rows[:counter], cols[:counter], data[:counter]


def calc_all_mz_correlations(intensity_matrix, min_spec_overlap_ratio=0.6, save=True, path=None):
    """
    Calculate m/z correlation matrix for MS imaging data

    :param intensity_matrix: 2D numpy array where rows are m/z values and columns are spectra
    :param min_spec_overlap_ratio: Minimum ratio of overlapping peaks between two spectra
    :param save: Boolean indicating whether to save the result
    :param path: Path to save the result if save is True
    :return: Sparse correlation matrix
    """
    n_mzs = intensity_matrix.shape[0]

    rows, cols, data = calc_mz_correlation_matrix_numba(intensity_matrix, min_spec_overlap_ratio)

    corr_matrix = csr_matrix((data, (rows, cols)), shape=(n_mzs, n_mzs), dtype=np.float64)
    corr_matrix = corr_matrix + corr_matrix.T
    corr_matrix.setdiag(1.0)

    if save and path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_npz(path, corr_matrix)

    return corr_matrix
