import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import multiprocessing as mp
from tqdm import tqdm
from _utils_imaging import PseudoMS1


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
    pseudo_ms1_spectra = _perform_clustering(mz_values, correlation_matrix,
                                             n_processes=n_processes,
                                             min_cluster_size=min_cluster_size)

    # Assign intensity values
    _assign_intensities(pseudo_ms1_spectra, intensity_matrix)

    if save and save_dir:
        save_path = os.path.join(save_dir, 'pseudo_ms1_spectra.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(pseudo_ms1_spectra, f)

    return pseudo_ms1_spectra


def _process_mz(args):
    """
    Process a single m/z value for clustering.
    """
    i, mz, mz_values, correlation_matrix, min_cluster_size = args

    # Get the row of the correlation matrix for the current m/z
    row = correlation_matrix[i].toarray().flatten()

    # Find all m/z values with non-zero correlation scores
    cluster_indices = np.nonzero(row)[0]

    if len(cluster_indices) >= min_cluster_size:
        # Form a pseudo MS1 spectrum
        cluster_mzs = mz_values[cluster_indices]

        # Store the actual indices in the original mz_values array
        indices = cluster_indices.tolist()

        return PseudoMS1(cluster_mzs.tolist(), [0] * len(cluster_mzs), indices)

    return None


def _perform_clustering(mz_values, correlation_matrix, n_processes=None,
                        min_cluster_size=6):
    """
    Perform clustering on m/z values based on correlation scores using multiprocessing.
    """
    # Ensure correlation_matrix is in CSR format for efficient row slicing
    if not isinstance(correlation_matrix, csr_matrix):
        correlation_matrix = csr_matrix(correlation_matrix)

    # Prepare arguments for multiprocessing
    args_list = [
        (i, mz, mz_values, correlation_matrix, min_cluster_size)
        for i, mz in enumerate(mz_values)]

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


def _assign_intensities(pseudo_ms1_spectra, intensity_matrix):
    """
    Assign intensity values to pseudo MS1 spectra.
    """
    for spectrum in tqdm(pseudo_ms1_spectra, desc="Assigning intensities"):
        # Get intensities for these indices
        intensities = intensity_matrix[spectrum.indices, :]

        # Find the spectrum with the highest total intensity for this cluster
        spectrum_sums = np.sum(intensities, axis=0)
        max_spectrum_index = np.argmax(spectrum_sums)

        # Select the intensities from the spectrum with the highest total intensity
        spectrum.intensities = intensities[:, max_spectrum_index].tolist()


def _remove_redundant_spectra(pseudo_ms1_spectra):
    """
    Remove redundant spectra (equal sets of m/z values).
    """
    non_redundant_spectra = []
    seen_mz_sets = set()

    for spec in tqdm(pseudo_ms1_spectra, desc="Removing redundant spectra"):
        mz_set = frozenset(spec.mzs)
        if mz_set not in seen_mz_sets:
            seen_mz_sets.add(mz_set)
            non_redundant_spectra.append(spec)

    return non_redundant_spectra
