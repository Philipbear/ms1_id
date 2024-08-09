import os
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, save_npz, load_npz
import multiprocessing as mp
import tempfile


@njit
def mz_correlation_numba(intensities1, intensities2, min_spec_overlap_ratio=0.9, min_overlap=5):
    x_non_zero_mask = intensities1 != 0
    y_non_zero_mask = intensities2 != 0

    non_zero_mask = (intensities1 != 0) & (intensities2 != 0)

    n = len(intensities1[non_zero_mask])
    smaller_length = min(len(intensities1[x_non_zero_mask]), len(intensities2[y_non_zero_mask]))
    if n < min_spec_overlap_ratio * smaller_length or n < min_overlap:
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

    return numerator / denominator if denominator != 0 else 0.0


def worker(start_idx, end_idx, mmap_filename, intensity_matrix_shape, min_spec_overlap_ratio, min_overlap,
           min_cor, return_dict):

    intensity_matrix = np.memmap(mmap_filename, dtype=np.float64, mode='r', shape=intensity_matrix_shape)
    n_mzs = intensity_matrix_shape[0]

    rows = []
    cols = []
    data = []

    for i in range(start_idx, end_idx):
        for j in range(i + 1, n_mzs):
            corr = mz_correlation_numba(intensity_matrix[i], intensity_matrix[j], min_spec_overlap_ratio, min_overlap)
            if corr >= min_cor:
                rows.append(i)
                cols.append(j)
                data.append(corr)

    return_dict[start_idx] = (rows, cols, data)


def calc_all_mz_correlations(intensity_matrix, min_spec_overlap_ratio=0.5, min_overlap=5,
                             min_cor=0.9,
                             save=True, save_dir=None, n_processes=None, chunk_size=1000):
    """
    Calculate m/z correlation matrix for MS imaging data using multiprocessing and numpy memmap

    :param intensity_matrix: 2D numpy array where rows are m/z values and columns are spectra
    :param min_spec_overlap_ratio: Minimum ratio of overlapping spectra between two ions
    :param min_overlap: Minimum number of overlapping spectra between two ions
    :param min_cor: Minimum correlation value to keep
    :param save: Boolean indicating whether to save the result
    :param save_dir: Directory to save the result if save is True
    :param n_processes: Number of processes to use (default: number of CPU cores)
    :param chunk_size: Number of rows to process in each chunk
    :return: Sparse correlation matrix
    """

    # check if result files exist
    if save_dir is not None:
        path = os.path.join(save_dir, 'mz_correlation_matrix.npz')
        if os.path.exists(path):
            print("Loading existing correlation matrix...")
            return load_npz(path)

    n_mzs, n_spectra = intensity_matrix.shape

    if n_processes is None:
        n_processes = mp.cpu_count()

    # Create a temporary memmap file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    mmap_filename = temp_file.name
    mmap_array = np.memmap(mmap_filename, dtype=np.float64, mode='w+', shape=intensity_matrix.shape)
    mmap_array[:] = intensity_matrix[:]
    mmap_array.flush()

    # Prepare chunks
    chunks = [(i, min(i + chunk_size, n_mzs)) for i in range(0, n_mzs, chunk_size)]

    print(f"Calculating correlations using {n_processes} processes...")
    manager = mp.Manager()
    return_dict = manager.dict()

    with mp.Pool(processes=n_processes) as pool:
        jobs = [
            pool.apply_async(worker, (start, end, mmap_filename, intensity_matrix.shape, min_spec_overlap_ratio,
                                      min_overlap, min_cor, return_dict))
            for start, end in chunks
        ]

        for job in jobs:
            job.get()  # Wait for the job to complete

    print("Combining results...")
    all_rows = []
    all_cols = []
    all_data = []
    for result in return_dict.values():
        all_rows.extend(result[0])
        all_cols.extend(result[1])
        all_data.extend(result[2])

    print("Creating sparse matrix...")
    corr_matrix = csr_matrix((all_data, (all_rows, all_cols)), shape=(n_mzs, n_mzs), dtype=np.float64)
    corr_matrix = corr_matrix + corr_matrix.T
    corr_matrix.setdiag(1.0)

    if save and save_dir:
        path = os.path.join(save_dir, 'mz_correlation_matrix.npz')
        print(f"Saving correlation matrix to {path}...")
        save_npz(path, corr_matrix)

    # Clean up the temporary memmap file
    os.unlink(mmap_filename)

    return corr_matrix


if __name__ == "__main__":
    import time
    start = time.time()
    intensity_matrix = np.random.rand(4000, 100)  # 4000 m/z values, 100 spectra

    # Calculate correlations
    corr_matrix = calc_all_mz_correlations(intensity_matrix, n_processes=10,
                                           save=False)

    print(f"Time elapsed: {time.time() - start:.2f} s")

    print(f"Correlation matrix shape: {corr_matrix.shape}")
