import os
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, save_npz
import multiprocessing as mp


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
def calc_mz_correlation_matrix_numba(intensity_matrix, start_idx, end_idx, min_spec_overlap_ratio=0.6):
    n_mzs = intensity_matrix.shape[0]
    max_correlations = (end_idx - start_idx) * n_mzs
    rows = np.zeros(max_correlations, dtype=np.int32)
    cols = np.zeros(max_correlations, dtype=np.int32)
    data = np.zeros(max_correlations, dtype=np.float64)
    counter = 0

    for i in range(start_idx, end_idx):
        for j in range(i + 1, n_mzs):
            corr = mz_correlation_numba(intensity_matrix[i], intensity_matrix[j], min_spec_overlap_ratio)
            rows[counter] = i
            cols[counter] = j
            data[counter] = corr
            counter += 1

    return rows[:counter], cols[:counter], data[:counter]


def worker(intensity_matrix, start_idx, end_idx, min_spec_overlap_ratio):
    return calc_mz_correlation_matrix_numba(intensity_matrix, start_idx, end_idx, min_spec_overlap_ratio)


def calc_all_mz_correlations(intensity_matrix, min_spec_overlap_ratio=0.6, save=True, save_dir=None, n_processes=None):
    """
    Calculate m/z correlation matrix for MS imaging data using multiprocessing

    :param intensity_matrix: 2D numpy array where rows are m/z values and columns are spectra
    :param min_spec_overlap_ratio: Minimum ratio of overlapping peaks between two spectra
    :param save: Boolean indicating whether to save the result
    :param save_dir: Directory to save the result if save is True
    :param n_processes: Number of processes to use (default: number of CPU cores)
    :return: Sparse correlation matrix
    """
    n_mzs = intensity_matrix.shape[0]

    if n_processes is None:
        n_processes = mp.cpu_count()

    chunk_size = n_mzs // n_processes
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_processes)]
    chunks[-1] = (chunks[-1][0], n_mzs)  # Ensure the last chunk goes to the end

    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(worker,
                               [(intensity_matrix, start, end, min_spec_overlap_ratio) for start, end in chunks])

    all_rows = np.concatenate([r[0] for r in results])
    all_cols = np.concatenate([r[1] for r in results])
    all_data = np.concatenate([r[2] for r in results])

    corr_matrix = csr_matrix((all_data, (all_rows, all_cols)), shape=(n_mzs, n_mzs), dtype=np.float64)
    corr_matrix = corr_matrix + corr_matrix.T
    corr_matrix.setdiag(1.0)

    if save and save_dir:
        path = os.path.join(save_dir, 'mz_correlation_matrix.npz')
        save_npz(path, corr_matrix)

    return corr_matrix


if __name__ == "__main__":
    # Create a sample intensity matrix
    intensity_matrix = np.random.rand(1000, 1000)  # 1000 m/z values, 1000 spectra

    # Calculate correlations
    corr_matrix = calc_all_mz_correlations(intensity_matrix, save=False)

    print(f"Correlation matrix shape: {corr_matrix.shape}")
