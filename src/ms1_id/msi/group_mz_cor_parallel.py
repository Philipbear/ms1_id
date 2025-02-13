import multiprocessing as mp
import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from ms1_id.msi.utils_imaging import PseudoMS2
from ms1_id.msi.export_msi import write_pseudoms2_to_mgf


def generate_pseudo_ms2(mz_values, intensity_matrix, correlation_matrix,
                        n_processes=None, min_cluster_size=6,
                        min_cor=0.90, max_cor_depth=1,
                        save=False, save_dir=None, chunk_size=1000):
    """
    Generate pseudo MS2 spectra for imaging data using chunked parallel processing
    """
    # Check if result files exist
    if save_dir:
        save_path = os.path.join(save_dir, 'pseudo_ms2_spectra.pkl')
        if os.path.exists(save_path):
            print("Loading existing pseudo MS2 spectra...")
            with open(save_path, 'rb') as f:
                return pickle.load(f)

    # Perform clustering
    pseudo_ms2_spectra = _perform_clustering(mz_values, correlation_matrix,
                                             n_processes=n_processes,
                                             min_cor=min_cor,
                                             max_cor_depth=max_cor_depth,
                                             min_cluster_size=min_cluster_size,
                                             chunk_size=chunk_size)

    # Assign intensity values
    _assign_intensities(pseudo_ms2_spectra, intensity_matrix)

    if save and save_dir:
        pkl_path = os.path.join(save_dir, 'pseudo_ms2_spectra.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(pseudo_ms2_spectra, f)

        mgf_path = os.path.join(save_dir, 'pseudo_ms2_spectra.mgf')
        write_pseudoms2_to_mgf(pseudo_ms2_spectra, mgf_path)

    return pseudo_ms2_spectra


def _process_chunk(args):
    """
    Process a chunk of m/z values for clustering, considering max_cor_depth and correlation threshold.
    """
    start_idx, end_idx, mz_values, correlation_matrix, min_cor, min_cluster_size, max_cor_depth = args
    chunk_results = []

    def find_correlated_mzs(target_idx, current_depth, visited):
        if current_depth > max_cor_depth:
            return set()

        row = correlation_matrix[target_idx].toarray().flatten()

        if current_depth == 0:
            # correlated_indices = set(np.where((mz_values <= mz_values[target_idx] + 1e-2) & (row > 0))[0]) - visited
            correlated_indices = set(np.where(row > 0)[0]) - visited
        else:
            # For depth >= 1, apply the correlation threshold with the original target
            original_target_row = correlation_matrix[start_idx].toarray().flatten()
            # correlated_indices = set(np.where((mz_values <= mz_values[target_idx] + 1e-2) &
            #                                   (row > 0) & (original_target_row >= (min_cor - 0.10)))[0]) - visited
            correlated_indices = set(np.where((row > 0) & (original_target_row >= (min_cor - 0.10)))[0]) - visited

        if current_depth < max_cor_depth:
            for idx in correlated_indices.copy():
                correlated_indices.update(find_correlated_mzs(idx, current_depth + 1, visited | correlated_indices))

        return correlated_indices

    for i in range(start_idx, end_idx):
        mz = mz_values[i]
        correlated_indices = find_correlated_mzs(i, 0, set())

        if len(correlated_indices) >= min_cluster_size:
            cluster_mzs = mz_values[list(correlated_indices)]
            indices = sorted(correlated_indices)
            chunk_results.append(PseudoMS2(mz, i, cluster_mzs.tolist(), [0] * len(cluster_mzs), indices))

    return chunk_results


def _perform_clustering(mz_values, correlation_matrix, n_processes=None, min_cor=0.90,
                        min_cluster_size=6, max_cor_depth=1, chunk_size=800):
    """
    Perform clustering on m/z values based on correlation scores using chunked multiprocessing.
    """
    if not isinstance(correlation_matrix, csr_matrix):
        correlation_matrix = csr_matrix(correlation_matrix)

    n_processes = n_processes or mp.cpu_count()

    # Prepare chunks
    n_chunks = (len(mz_values) + chunk_size - 1) // chunk_size
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, len(mz_values)),
               mz_values, correlation_matrix, min_cor, min_cluster_size, max_cor_depth)
              for i in range(n_chunks)]

    # Process chunks in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(_process_chunk, chunks),
                            total=len(chunks), desc="Processing chunks"))

    # Flatten results
    pseudo_ms2_spectra = [spectrum for chunk_result in results for spectrum in chunk_result]

    return pseudo_ms2_spectra


def _assign_intensities(pseudo_ms2_spectra, intensity_matrix):
    """
    Assign intensity values to pseudo MS2 spectra.
    """
    for spectrum in tqdm(pseudo_ms2_spectra, desc="Assigning intensities"):
        # Get the intensities for all m/z values in this PseudoMS1 object
        intensities = intensity_matrix[spectrum.indices, :]

        # Get the intensities for the target m/z across all spectra
        t_mz_intensities = intensity_matrix[spectrum.t_mz_idx, :]

        # Find the spectrum with the highest intensity at the target m/z
        max_spectrum_index = np.argmax(t_mz_intensities)

        spectrum.intensities = intensities[:, max_spectrum_index].tolist()
