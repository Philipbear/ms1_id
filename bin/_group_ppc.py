"""
This is to get all features with high PPCs, generate pseudo MS1 for a single file
"""
import os
import pickle
from collections import defaultdict

import numpy as np
from numba import njit

import networkx as nx
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
        if file.endswith('_pseudoMS1.pkl'):
            try:
                with open(os.path.join(config.single_file_dir, file), 'rb') as f:
                    new_pseudo_ms1_spectra = pickle.load(f)
                    pseudo_ms1_spectra.extend(new_pseudo_ms1_spectra)
            except:
                continue

    return pseudo_ms1_spectra


def generate_pseudo_ms1(msdata, ppc_matrix, peak_cor_rt_tol=0.1,
                        min_ppc=0.7, resolution=1.5,
                        save=False, save_dir=None):
    """
    Generate pseudo MS1 spectra for a single file
    :param msdata: MSData object
    :param ppc_matrix: sparse matrix of PPC scores
    :param peak_cor_rt_tol: peak correlation retention time tolerance
    :param min_ppc: minimum PPC threshold for overlapping communities
    :param resolution: resolution parameter for Louvain method
        (Lower resolution values (< 1.0) tend to result in fewer, larger communities.)
    :param save: whether to save the pseudo MS1 spectra
    :param save_dir: directory to save the pseudo MS1 spectra
    :return:
    """
    roi_pairs = _gen_ppc_for_rois(msdata, ppc_matrix, rt_tol=peak_cor_rt_tol)

    cluster_rois = _perform_louvain_for_roi_pairs(roi_pairs, min_ppc=min_ppc, resolution=resolution)

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

    roi_pairs = []
    for i, j in valid_pairs:
        roi_a = rois[i]
        roi_b = rois[j]
        ppc = ppc_matrix[roi_id_to_index[roi_a.id], roi_id_to_index[roi_b.id]]
        roi_pairs.append(RoiPair(roi_a, roi_b, ppc))

    return roi_pairs


def refine_clusters_by_rt(cluster_rois, roi_dict, peak_cor_rt_tol):
    """
    Refine clusters by ensuring the RT range in each cluster is within peak_cor_rt_tol

    :param cluster_rois: dictionary of cluster labels and ROI IDs
    :param roi_dict: dictionary of ROI ID to ROI object
    :param peak_cor_rt_tol: maximum allowed RT difference within a cluster
    :return: refined dictionary of cluster labels and ROI IDs
    """
    refined_clusters = {}

    for cluster_label, roi_ids in cluster_rois.items():
        if len(roi_ids) < 2:
            refined_clusters[cluster_label] = roi_ids
            continue

        # Sort ROIs by retention time
        sorted_rois = sorted(roi_ids, key=lambda roi_id: roi_dict[roi_id].rt)

        # Find the median RT
        median_rt = np.median([roi_dict[roi_id].rt for roi_id in sorted_rois])

        # Initialize the refined cluster with ROIs within tolerance of the median RT
        refined_cluster = []
        for roi_id in sorted_rois:
            if abs(roi_dict[roi_id].rt - median_rt) <= peak_cor_rt_tol / 2:
                refined_cluster.append(roi_id)

        # If we have at least 2 ROIs in the refined cluster, add it to refined_clusters
        if len(refined_cluster) >= 2:
            refined_clusters[cluster_label] = set(refined_cluster)

    print(f"Cluster refinement summary:")
    print(f"  Original number of clusters: {len(cluster_rois)}")
    print(f"  Total ROIs in original clusters: {sum(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Refined number of clusters: {len(refined_clusters)}")
    print(f"  Total ROIs in refined clusters: {sum(len(cluster) for cluster in refined_clusters.values())}")

    return refined_clusters


def _perform_louvain_for_roi_pairs(roi_pairs, min_ppc=0.7, resolution=1.0, min_cluster_size=6, peak_cor_rt_tol=0.1):
    """
    Cluster ROIs allowing for overlap based on high PPC scores
    :param roi_pairs: list of RoiPair objects
    :param min_ppc: min PPC score for clustering
    :param resolution: resolution parameter for Louvain clustering (default 1.0)
    :param min_cluster_size: minimum number of ROIs in a cluster
    :param peak_cor_rt_tol: peak correlation retention time tolerance
    :return: dictionary of cluster labels and ROI IDs
    """
    # Create a graph with edges for high PPC scores only
    G = nx.Graph()
    for rp in roi_pairs:
        if rp.ppc >= min_ppc:
            G.add_edge(rp.roi_1.id, rp.roi_2.id, weight=rp.ppc)

    # Perform Louvain clustering
    partition = nx.community.louvain_communities(G, weight='weight', resolution=resolution)

    # Convert partition to cluster dictionary and filter small clusters
    cluster_rois = {i: set(cluster) for i, cluster in enumerate(partition) if len(cluster) >= min_cluster_size}

    # Create a dictionary of ROI objects
    roi_dict = {}
    for rp in roi_pairs:
        roi_dict[rp.roi_1.id] = rp.roi_1
        roi_dict[rp.roi_2.id] = rp.roi_2

    # Refine clusters based on RT
    refined_cluster_rois = refine_clusters_by_rt(cluster_rois, roi_dict, peak_cor_rt_tol)

    # Print summary
    total_rois = sum(len(cluster) for cluster in refined_cluster_rois.values())
    avg_cluster_size = total_rois / len(refined_cluster_rois) if refined_cluster_rois else 0

    print(f"  Average cluster size: {avg_cluster_size:.2f}")

    return refined_cluster_rois


def _map_cluster_labels_to_pseudo_ms1(msdata, cluster_rois):
    """
    Map clustering labels to PseudoMS1 objects
    :param msdata: MSData object
    :param cluster_rois: dictionary of cluster labels and ROI IDs
    :return: list of PseudoMS1 objects
    """
    pseudo_ms1_spectra = []
    roi_dict = {roi.id: roi for roi in msdata.rois}

    for cluster_label, roi_ids in cluster_rois.items():
        mz_ls, rt_ls, int_ls = [], [], []

        for roi_id in roi_ids:
            if roi_id in roi_dict:
                roi = roi_dict[roi_id]
                mz_ls.append(roi.mz)
                rt_ls.append(roi.rt)
                int_ls.append(roi.peak_height)

        if len(rt_ls) > 0:  # Only create PseudoMS1 if the cluster is not empty
            avg_rt = sum(rt_ls) / len(rt_ls)
            pseudo_ms1 = PseudoMS1(mz_ls, int_ls, list(roi_ids), msdata.file_name, avg_rt)
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
