"""
This is to get all features with high PPCs, generate pseudo MS1 for a single file
"""
from collections import defaultdict
import numpy as np
import os
from numba import njit
import hdbscan
import pickle
from scipy.sparse import lil_matrix, csr_matrix

from _utils import PseudoMS1, RoiPair


def retrieve_pseudo_ms1_spectra(config):
    """
    Retrieve pseudo MS1 spectra for all files
    :param config: Config object
    :return: dictionary of pseudo MS1 spectra
    """
    pseudo_ms1_spectra = []

    files = os.listdir(config.single_file_dir)
    for file in files:
        if file.endswith('_PseudoMS1.pkl'):
            try:
                with open(os.path.join(config.single_file_dir, file), 'rb') as f:
                    new_pseudo_ms1_spectra = pickle.load(f)
                    pseudo_ms1_spectra.extend(new_pseudo_ms1_spectra)
            except:
                continue

    return pseudo_ms1_spectra


def generate_pseudo_ms1(msdata, ppc_matrix, peak_cor_rt_tol=0.1, save=False, save_dir=None):
    """
    Generate pseudo MS1 spectra for a single file
    :param msdata: MSData object
    :param ppc_matrix: sparse matrix of PPC scores
    :param peak_cor_rt_tol: peak correlation retention time tolerance
    :param save: whether to save the pseudo MS1 spectra
    :param save_dir: directory to save the pseudo MS1 spectra
    :return:
    """
    roi_pairs = _gen_ppc_for_rois(msdata, ppc_matrix, rt_tol=peak_cor_rt_tol)
    cluster_rois = _perform_hdbscan_for_roi_pairs(roi_pairs)
    pseudo_ms1_spectra = _map_cluster_labels_to_pseudo_ms1(msdata, cluster_rois)

    if save:
        if save_dir is None:
            path = os.path.join(msdata.params.single_file_dir, msdata.file_name + "_pseudoMS1.pkl")
        else:
            path = os.path.splitext(save_dir)[0] + "_pseudoMS1.pkl"

        with open(path, 'wb') as f:
            pickle.dump(pseudo_ms1_spectra, f)

    return pseudo_ms1_spectra


def _gen_ppc_for_rois(msdata, ppc_matrix, rt_tol=0.1):
    """
    Generate PPC scores for ROI pairs
    """

    rois = msdata.rois
    rois.sort(key=lambda x: x.id)

    # Create a mapping of ROI IDs to their indices
    roi_id_to_index = {roi.id: i for i, roi in enumerate(rois)}

    ids = np.array([roi.id for roi in rois])
    rts = np.array([roi.rt for roi in rois])

    valid_pairs = find_valid_pairs(ids, rts, rt_tol)

    roi_pairs = [RoiPair(ids[i], ids[j]) for i, j in valid_pairs]

    # Assign PPC scores to RoiPair objects using the mapping
    for rp in roi_pairs:
        i, j = roi_id_to_index[rp.id_1], roi_id_to_index[rp.id_2]
        rp.ppc = ppc_matrix[i, j]

    return roi_pairs


def _perform_hdbscan_for_roi_pairs(roi_pairs, min_cluster_size=5, min_samples=None):
    """
    Perform HDBSCAN clustering on RoiPair objects
    :param roi_pairs: list of RoiPair objects
    :param min_cluster_size: minimum cluster size for HDBSCAN
    :param min_samples: minimum number of samples for HDBSCAN
    :return: cluster ROIs
    """
    unique_ids = list(set([rp.id_1 for rp in roi_pairs] + [rp.id_2 for rp in roi_pairs]))
    id_to_index = {id_: index for index, id_ in enumerate(unique_ids)}
    index_to_id = {index: id_ for id_, index in id_to_index.items()}
    num_rois = len(unique_ids)

    similarity_matrix = np.zeros((num_rois, num_rois))
    for rp in roi_pairs:
        index_1, index_2 = id_to_index[rp.id_1], id_to_index[rp.id_2]
        similarity_matrix[index_1, index_2] = similarity_matrix[index_2, index_1] = rp.ppc

    distance_matrix = 1 - similarity_matrix

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(metric='precomputed',
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples)
    clusterer.fit(distance_matrix)

    # Use probabilities as a proxy for soft clustering
    cluster_rois = defaultdict(set)
    for index, (label, prob) in enumerate(zip(clusterer.labels_, clusterer.probabilities_)):
        roi_id = index_to_id[index]
        if label != -1:
            cluster_rois[label].add(roi_id)
        else:
            # Assign to the cluster with highest probability among neighboring points
            neighbor_indices = np.where(distance_matrix[index] < np.median(distance_matrix[index]))[0]
            neighbor_labels = clusterer.labels_[neighbor_indices]
            if len(neighbor_labels) > 0 and not np.all(neighbor_labels == -1):
                valid_labels = neighbor_labels[neighbor_labels != -1]
                if len(valid_labels) > 0:
                    most_common_label = np.argmax(np.bincount(valid_labels))
                    cluster_rois[most_common_label].add(roi_id)

    return cluster_rois


def _map_cluster_labels_to_pseudo_ms1(msdata, cluster_rois):
    """
    Map clustering labels to PseudoMS1 objects
    :param msdata: MSData object
    :param cluster_rois: dictionary of cluster labels and ROI IDs
    :return: list of PseudoMS1 objects
    """
    pseudo_ms1_spectra = []
    roi_dict = {roi.id: roi for roi in msdata.rois}

    for cluster_label in cluster_rois.keys():
        roi_ids = list(cluster_rois[cluster_label])
        mz_ls, rt_ls, int_ls = [], [], []

        for roi_id in roi_ids:
            roi = roi_dict[roi_id]
            mz_ls.append(roi.mz)
            rt_ls.append(roi.rt)
            int_ls.append(roi.peak_height)

        avg_rt = sum(rt_ls) / len(rt_ls)

        pseudo_ms1 = PseudoMS1(mz_ls, int_ls, roi_ids, msdata.file_name, avg_rt)
        pseudo_ms1_spectra.append(pseudo_ms1)

    return pseudo_ms1_spectra


@njit
def find_valid_pairs(ids, rts, rt_tol):
    n = len(ids)
    valid_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(rts[i] - rts[j]) <= rt_tol:
                valid_pairs.append((i, j))
    return valid_pairs
