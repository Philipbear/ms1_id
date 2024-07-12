"""
This is to get all features with high PPCs, generate pseudo MS1 for a single file
"""
from collections import defaultdict
import numpy as np
import hdbscan
from scipy.sparse import lil_matrix, csr_matrix

from _utils import PseudoMS1


def generate_pseudo_ms1_single(msdata, ppc_matrix, peak_cor_rt_tol=0.1, hdbscan_prob_cutoff=0.2):
    """
    Generate PseudoMS1 spectra for a single file
    :param msdata: MSData object
    :param ppc_matrix: sparse matrix of PPC scores
    :param peak_cor_rt_tol: retention time tolerance for aligning features
    :param hdbscan_prob_cutoff: probability cutoff for HDBSCAN clustering
    :return: list of PseudoMS1 objects
    """
    # Filter PPC matrix based on RT tolerance
    filtered_ppc_matrix = _filter_ppc_matrix_by_rt(msdata, ppc_matrix, peak_cor_rt_tol)

    all_cluster_labels, cluster_features = _perform_hdbscan(filtered_ppc_matrix, hdbscan_prob_cutoff)

    pseudo_ms1_spectra = []
    roi_dict = {roi.id: roi for roi in msdata.rois}

    for cluster_label in all_cluster_labels:
        feature_ids = list(cluster_features[cluster_label])
        mz_ls, rt_ls, int_ls = [], [], []

        for feature_id in feature_ids:
            roi = roi_dict[feature_id]
            mz_ls.append(roi.mz)
            rt_ls.append(roi.rt)
            int_ls.append(roi.intensity)

        avg_rt = sum(rt_ls) / len(rt_ls)

        pseudo_ms1 = PseudoMS1(mz_ls, int_ls, feature_ids, avg_rt)
        pseudo_ms1_spectra.append(pseudo_ms1)

    return pseudo_ms1_spectra


def _filter_ppc_matrix_by_rt(msdata, ppc_matrix, rt_tol):
    """
    Filter PPC matrix based on retention time tolerance
    :param msdata: MSData object
    :param ppc_matrix: sparse matrix of PPC scores
    :param rt_tol: retention time tolerance for aligning features
    """
    rois = msdata.rois
    rt_array = np.array([roi.rt for roi in rois])

    rt_diff = np.abs(rt_array[:, np.newaxis] - rt_array)
    rt_mask = rt_diff <= rt_tol

    # Convert to LIL format for efficient modification
    lil_ppc_matrix = ppc_matrix.tolil()

    # Apply the RT filter
    lil_ppc_matrix[~rt_mask] = 0

    # Convert back to CSR format for efficient computations
    filtered_matrix = lil_ppc_matrix.tocsr()

    return filtered_matrix


def _perform_hdbscan(ppc_matrix, prob_cutoff=0.2, min_cluster_size=5, min_samples=None):
    """
    Perform HDBSCAN clustering using PPC matrix
    :param ppc_matrix: sparse matrix of PPC scores
    :param prob_cutoff: probability cutoff for HDBSCAN clustering
    :param min_cluster_size: minimum cluster size for HDBSCAN
    :param min_samples: minimum number of samples for HDBSCAN
    :return: all cluster labels, cluster features
    """
    distance_matrix = 1 - ppc_matrix.toarray()

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(metric='precomputed',
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples)
    clusterer.fit(distance_matrix)

    all_cluster_labels = set(label for label in clusterer.labels_ if label != -1)

    # Use probabilities as a proxy for soft clustering
    cluster_features = defaultdict(set)
    for index, (label, prob) in enumerate(zip(clusterer.labels_, clusterer.probabilities_)):
        if label != -1 and prob >= prob_cutoff:
            cluster_features[label].add(index)
        else:
            # Assign to the cluster with highest probability among neighboring points
            neighbor_labels = clusterer.labels_[distance_matrix[index] <= np.median(distance_matrix[index])]
            if len(neighbor_labels) > 0:
                most_common_label = np.argmax(np.bincount(neighbor_labels[neighbor_labels != -1]))
                cluster_features[most_common_label].add(index)

    return all_cluster_labels, cluster_features