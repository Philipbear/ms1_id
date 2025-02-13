import multiprocessing as mp
import os
import tempfile

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm import tqdm


@njit
def _mz_correlation(intensities1, intensities2, min_overlap=5):
    # Remove zero intensities
    non_zero_mask = (intensities1 != 0) & (intensities2 != 0)
    x = intensities1[non_zero_mask]
    y = intensities2[non_zero_mask]

    n = len(x)
    if n < min_overlap:
        return 0.0

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    sum_y2 = np.sum(y * y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

    return numerator / denominator if denominator != 0 else 0.0


def worker(start_idx, end_idx, mmap_filename, intensity_matrix_shape, min_overlap,
           min_cor, return_dict):
    intensity_matrix = np.memmap(mmap_filename, dtype=np.float64, mode='r', shape=intensity_matrix_shape)
    n_mzs = intensity_matrix_shape[0]

    rows = []
    cols = []
    data = []

    for i in range(start_idx, end_idx):
        for j in range(i + 1, n_mzs):
            corr = _mz_correlation(intensity_matrix[i], intensity_matrix[j], min_overlap)
            if corr >= min_cor:
                rows.append(i)
                cols.append(j)
                data.append(corr)

    return_dict[start_idx] = (rows, cols, data)


def calc_all_mz_correlations(intensity_matrix, min_overlap=5, min_cor=0.8,
                             save=True, save_dir=None, n_processes=None, chunk_size=500):
    """
    Calculate m/z correlation matrix for MS imaging data using multiprocessing and numpy memmap

    :param intensity_matrix: 2D numpy array where rows are m/z values and columns are spectra
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

    n_processes = n_processes or mp.cpu_count()

    # Create a temporary memmap file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    mmap_filename = temp_file.name
    mmap_array = np.memmap(mmap_filename, dtype=np.float64, mode='w+', shape=intensity_matrix.shape)
    mmap_array[:] = intensity_matrix[:]
    mmap_array.flush()

    # Prepare chunks
    chunks = [(i, min(i + chunk_size, n_mzs)) for i in range(0, n_mzs, chunk_size)]

    print(f"Calculating m/z spatial correlations...")

    if n_processes == 1:
        # Non-parallel processing
        all_rows = []
        all_cols = []
        all_data = []
        for start, end in tqdm(chunks, desc="Processing chunks"):
            rows, cols, data = [], [], []
            for i in range(start, end):
                for j in range(i + 1, n_mzs):
                    corr = _mz_correlation(mmap_array[i], mmap_array[j], min_overlap)
                    if corr >= min_cor:
                        rows.append(i)
                        cols.append(j)
                        data.append(corr)
            all_rows.extend(rows)
            all_cols.extend(cols)
            all_data.extend(data)
    else:
        # Parallel processing
        manager = mp.Manager()
        return_dict = manager.dict()

        with mp.Pool(processes=n_processes) as pool:
            jobs = [
                pool.apply_async(worker, (start, end, mmap_filename, intensity_matrix.shape,
                                          min_overlap, min_cor, return_dict))
                for start, end in chunks
            ]

            for job in tqdm(jobs, desc="Processing chunks"):
                job.get()  # Wait for the job to complete

        all_rows = []
        all_cols = []
        all_data = []
        for result in return_dict.values():
            all_rows.extend(result[0])
            all_cols.extend(result[1])
            all_data.extend(result[2])

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
