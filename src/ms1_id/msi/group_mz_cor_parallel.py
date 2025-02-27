import multiprocessing as mp
import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from ms1_id.msi.utils_imaging import PseudoMS2
from ms1_id.msi.export_msi import write_pseudoms2_to_mgf


def generate_pseudo_ms2(feature_ls, intensity_matrix, correlation_matrix,
                        n_processes=None,
                        min_cluster_size=6,
                        min_cor=0.90,
                        save_dir=None,
                        chunk_size=1000):
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
    pseudo_ms2_spectra = _perform_clustering(feature_ls, correlation_matrix,
                                             n_processes=n_processes,
                                             min_cor=min_cor,
                                             min_cluster_size=min_cluster_size,
                                             chunk_size=chunk_size)

    # Assign intensity values
    _assign_intensities(pseudo_ms2_spectra, intensity_matrix)

    # sort ms2_spec_ls by t_mz, add spec_idx
    pseudo_ms2_spectra = sorted(pseudo_ms2_spectra, key=lambda x: x.t_mz)
    for idx, spec in enumerate(pseudo_ms2_spectra):
        spec.spec_idx = idx

    if save_dir:
        pkl_path = os.path.join(save_dir, 'pseudo_ms2_spectra.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(pseudo_ms2_spectra, f)

        mgf_path = os.path.join(save_dir, 'pseudo_ms2_spectra.mgf')
        write_pseudoms2_to_mgf(pseudo_ms2_spectra, mgf_path)

    return pseudo_ms2_spectra


def _process_chunk(args):
    """
    Process a chunk of m/z values for clustering, considering correlation threshold.
    """
    start_idx, end_idx, mz_values, correlation_matrix, min_cor, min_cluster_size = args
    chunk_results = []

    for i in range(start_idx, end_idx):
        mz = mz_values[i]

        # Get correlated indices directly from the correlation matrix row
        row = correlation_matrix[i].toarray().flatten()
        correlated_indices = set(np.where(row > 0)[0])

        if len(correlated_indices) >= min_cluster_size:
            # Extract cluster m/z values and sort them
            cluster_mzs = np.sort(mz_values[list(correlated_indices)]).tolist()

            # Sort indices
            indices = sorted(correlated_indices)

            chunk_results.append(PseudoMS2(mz, cluster_mzs, [0] * len(cluster_mzs), indices))

    return chunk_results


def _perform_clustering(feature_ls, correlation_matrix, n_processes=None, min_cor=0.90,
                        min_cluster_size=3, chunk_size=800):
    """
    Perform clustering on m/z values based on correlation scores using chunked multiprocessing.
    """
    if not isinstance(correlation_matrix, csr_matrix):
        correlation_matrix = csr_matrix(correlation_matrix)

    mz_values = np.array([feature.mz for feature in feature_ls])

    # Prepare chunks
    n_chunks = (len(mz_values) + chunk_size - 1) // chunk_size
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, len(mz_values)),
               mz_values, correlation_matrix, min_cor, min_cluster_size)
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
    For each spectrum, find the scan where the most m/z values show up
    (have intensity > 0).
    """
    for spectrum in tqdm(pseudo_ms2_spectra, desc="Assigning intensities"):
        # Get the intensities for all m/z values in this PseudoMS2 object
        intensities = intensity_matrix[spectrum.indices, :]

        # Count number of non-zero intensities for each scan
        non_zero_counts = np.count_nonzero(intensities > 0, axis=0)

        # Find the scan with the most non-zero intensities
        max_spectrum_index = np.argmax(non_zero_counts)

        # Assign the intensities from the scan with most non-zero values
        spectrum.intensities = intensities[:, max_spectrum_index].tolist()

        ##################################################
        # Get the scan with the highest total intensity
        # # Sum intensities across all m/z values for each scan
        # total_intensities_per_scan = np.sum(intensities, axis=0)
        #
        # # Find the scan with the highest total intensity
        # max_spectrum_index = np.argmax(total_intensities_per_scan)
        #
        # # Assign the intensities from the scan with highest total intensity
        # spectrum.intensities = intensities[:, max_spectrum_index].tolist()