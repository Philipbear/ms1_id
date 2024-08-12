import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import multiprocessing as mp
from tqdm import tqdm
from _utils_imaging import PseudoMS1


def generate_pseudo_ms1(mz_values, intensity_matrix, correlation_matrix,
                        n_processes=None, min_cluster_size=6,
                        save=False, save_dir=None, chunk_size=1000):
    """
    Generate pseudo MS1 spectra for imaging data using chunked parallel processing
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
                                             min_cluster_size=min_cluster_size,
                                             chunk_size=chunk_size)

    # Assign intensity values
    _assign_intensities(pseudo_ms1_spectra, intensity_matrix)

    if save and save_dir:
        save_path = os.path.join(save_dir, 'pseudo_ms1_spectra.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(pseudo_ms1_spectra, f)

    return pseudo_ms1_spectra


def _process_chunk(args):
    """
    Process a chunk of m/z values for clustering.
    """
    start_idx, end_idx, mz_values, correlation_matrix, min_cluster_size = args
    chunk_results = []

    for i in range(start_idx, end_idx):
        mz = mz_values[i]
        row = correlation_matrix[i].toarray().flatten()

        # Only consider m/z values smaller than the target m/z
        smaller_mz_indices = np.where((mz_values <= mz + 1e-3) & (row > 0))[0]

        if len(smaller_mz_indices) >= min_cluster_size:
            cluster_mzs = mz_values[smaller_mz_indices]
            indices = smaller_mz_indices.tolist()
            chunk_results.append(PseudoMS1(mz, i, cluster_mzs.tolist(), [0] * len(cluster_mzs), indices))

    return chunk_results


def _perform_clustering(mz_values, correlation_matrix, n_processes=None,
                        min_cluster_size=6, chunk_size=800):
    """
    Perform clustering on m/z values based on correlation scores using chunked multiprocessing.
    """
    if not isinstance(correlation_matrix, csr_matrix):
        correlation_matrix = csr_matrix(correlation_matrix)

    if n_processes is None:
        n_processes = mp.cpu_count()

    # Prepare chunks
    n_chunks = (len(mz_values) + chunk_size - 1) // chunk_size
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, len(mz_values)),
               mz_values, correlation_matrix, min_cluster_size)
              for i in range(n_chunks)]

    # Process chunks in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(_process_chunk, chunks),
                            total=len(chunks), desc="Processing chunks"))

    # Flatten results
    pseudo_ms1_spectra = [spectrum for chunk_result in results for spectrum in chunk_result]

    # Remove redundant spectra
    non_redundant_spectra = _remove_redundant_spectra(pseudo_ms1_spectra)

    return non_redundant_spectra


def _assign_intensities(pseudo_ms1_spectra, intensity_matrix):
    """
    Assign intensity values to pseudo MS1 spectra.
    """
    for spectrum in tqdm(pseudo_ms1_spectra, desc="Assigning intensities"):
        # Get the intensities for all m/z values in this PseudoMS1 object
        intensities = intensity_matrix[spectrum.indices, :]

        # Get the intensities for the target m/z across all spectra
        t_mz_intensities = intensity_matrix[spectrum.t_mz_idx, :]

        # Find the spectrum with the highest intensity at the target m/z
        max_spectrum_index = np.argmax(t_mz_intensities)

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
