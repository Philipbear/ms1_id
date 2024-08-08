import os
import pickle
import numpy as np
from _utils_imaging import PseudoMS1
from scipy.sparse import csr_matrix
import multiprocessing as mp
from tqdm import tqdm


def generate_pseudo_ms1(mz_values, intensity_matrix, correlation_matrix,
                        n_processes=None, min_cluster_size=6,
                        save=False, save_dir=None):
    """
    Generate pseudo MS1 spectra for imaging data
    """

    # Check if result files exist
    if save_dir:
        save_path = os.path.join(save_dir, 'pseudo_ms1_spectra.pkl')
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                return pickle.load(f)

    # Perform clustering
    pseudo_ms1_spectra = _perform_clustering(mz_values, intensity_matrix, correlation_matrix,
                                             n_processes=n_processes,
                                             min_cluster_size=min_cluster_size)

    if save and save_dir:
        save_path = os.path.join(save_dir, 'pseudo_ms1_spectra.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(pseudo_ms1_spectra, f)

    return pseudo_ms1_spectra


def _process_mz(args):
    """
    Process a single m/z value for clustering.
    """
    i, mz, sorted_indices, mz_values, intensity_matrix, correlation_matrix, min_cluster_size = args

    # Get the row of the correlation matrix for the current m/z
    row = correlation_matrix[sorted_indices[i]].toarray().flatten()

    # Find all m/z values with non-zero correlation scores
    cluster_indices = np.nonzero(row)[0]

    if len(cluster_indices) >= min_cluster_size:
        # Form a pseudo MS1 spectrum
        cluster_mzs = mz_values[cluster_indices]
        cluster_intensities = intensity_matrix[cluster_indices, :]

        # Find the spectrum with the highest total intensity for this cluster
        spectrum_sums = np.sum(cluster_intensities, axis=0)
        max_spectrum_index = np.argmax(spectrum_sums)

        # Select the intensities from the spectrum with the highest total intensity
        int_ls = cluster_intensities[:, max_spectrum_index].tolist()

        return PseudoMS1(cluster_mzs.tolist(), int_ls)

    return None


def _perform_clustering(mz_values, intensity_matrix, correlation_matrix, n_processes=None,
                        min_cluster_size=6):
    """
    Perform clustering on m/z values based on correlation scores using multiprocessing.
    """
    # Ensure correlation_matrix is in CSR format for efficient row slicing
    if not isinstance(correlation_matrix, csr_matrix):
        correlation_matrix = csr_matrix(correlation_matrix)

    # Sort m/z values
    sorted_indices = np.argsort(mz_values)
    sorted_mz_values = mz_values[sorted_indices]

    # Prepare arguments for multiprocessing
    args_list = [
        (i, mz, sorted_indices, mz_values, intensity_matrix, correlation_matrix, min_cluster_size)
        for i, mz in enumerate(sorted_mz_values)]

    # Use multiprocessing to process m/z values in parallel
    if n_processes is None:
        n_processes = mp.cpu_count()

    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(_process_mz, args_list), total=len(args_list), desc="Clustering"))

    # Filter out None results and create the pseudo_ms1_spectra list
    pseudo_ms1_spectra = [result for result in results if result is not None]

    # Remove redundant spectra
    non_redundant_spectra = _remove_redundant_spectra(pseudo_ms1_spectra)

    return non_redundant_spectra


def _remove_redundant_spectra(pseudo_ms1_spectra):
    """
    Remove redundant spectra (subsets of larger spectra).
    """
    non_redundant_spectra = []
    for i, spec1 in enumerate(tqdm(pseudo_ms1_spectra, desc="Removing redundant spectra")):
        is_subset = False
        for j, spec2 in enumerate(pseudo_ms1_spectra):
            if i != j and set(spec1.mzs).issubset(set(spec2.mzs)):
                is_subset = True
                break
        if not is_subset:
            non_redundant_spectra.append(spec1)
    return non_redundant_spectra
