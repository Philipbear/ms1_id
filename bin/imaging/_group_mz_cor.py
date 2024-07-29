import os
import pickle
import numpy as np
from _utils_imaging import PseudoMS1
from scipy.sparse import csr_matrix


def generate_pseudo_ms1(mz_values, intensity_matrix, correlation_matrix,
                        min_correlation=0.8, min_cluster_size=6,
                        save=False, save_dir=None):
    """
    Generate pseudo MS1 spectra for imaging data
    """
    # Perform clustering
    pseudo_ms1_spectra = _perform_clustering(mz_values, intensity_matrix, correlation_matrix,
                                             min_correlation=min_correlation,
                                             min_cluster_size=min_cluster_size)

    if save and save_dir:
        save_path = os.path.join(save_dir, 'pseudo_ms1_spectra.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(pseudo_ms1_spectra, f)

    return pseudo_ms1_spectra


def _perform_clustering(mz_values, intensity_matrix, correlation_matrix,
                        min_correlation=0.8, min_cluster_size=6):
    """
    Perform clustering on m/z values based on correlation scores.
    """
    # Ensure correlation_matrix is in CSR format for efficient row slicing
    if not isinstance(correlation_matrix, csr_matrix):
        correlation_matrix = csr_matrix(correlation_matrix)

    # Sort m/z values
    sorted_indices = np.argsort(mz_values)
    sorted_mz_values = mz_values[sorted_indices]

    pseudo_ms1_spectra = []

    for i, mz in enumerate(sorted_mz_values):
        # Get the row of the correlation matrix for the current m/z
        row = correlation_matrix[sorted_indices[i]].toarray().flatten()

        # Find all m/z values with correlation scores above the threshold
        cluster_indices = np.where(row >= min_correlation)[0]

        if len(cluster_indices) >= min_cluster_size:
            # Form a pseudo MS1 spectrum
            cluster_mzs = mz_values[cluster_indices]
            cluster_intensities = intensity_matrix[cluster_indices, :]

            # Find the spectrum with the highest total intensity for this cluster
            spectrum_sums = np.sum(cluster_intensities, axis=0)
            max_spectrum_index = np.argmax(spectrum_sums)

            # Select the intensities from the spectrum with the highest total intensity
            int_ls = cluster_intensities[:, max_spectrum_index].tolist()

            pseudo_ms1 = PseudoMS1(cluster_mzs.tolist(), int_ls)
            pseudo_ms1_spectra.append(pseudo_ms1)

    # Remove redundant spectra
    non_redundant_spectra = _remove_redundant_spectra(pseudo_ms1_spectra)

    return non_redundant_spectra


def _remove_redundant_spectra(pseudo_ms1_spectra):
    """
    Remove redundant spectra (subsets of larger spectra).
    """
    non_redundant_spectra = []
    for i, spec1 in enumerate(pseudo_ms1_spectra):
        is_subset = False
        for j, spec2 in enumerate(pseudo_ms1_spectra):
            if i != j and set(spec1.mzs).issubset(set(spec2.mzs)):
                is_subset = True
                break
        if not is_subset:
            non_redundant_spectra.append(spec1)
    return non_redundant_spectra

