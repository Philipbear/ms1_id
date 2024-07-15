"""
This is to get all features with high PPCs, generate pseudo MS1 for a single file
"""
import os
import pickle
from collections import defaultdict

import networkx as nx

from _utils import PseudoMS1


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


def generate_pseudo_ms1(msdata, ppc_matrix, peak_cor_rt_tol=0.05, min_ppc=0.8, min_cluster_size=6,
                        min_overlap_ppc=0.95, save=False, save_dir=None):
    """
    Generate pseudo MS1 spectra for a single file
    :param msdata: MSData object
    :param ppc_matrix: sparse matrix of PPC scores
    :param peak_cor_rt_tol: RT tolerance for clustering
    :param min_ppc: minimum PPC score for clustering
    :param min_overlap_ppc: minimum PPC score for allowing overlaps
    :param min_cluster_size: minimum number of ROIs in a cluster
    :param save: whether to save the pseudo MS1 spectra
    :param save_dir: directory to save the pseudo MS1 spectra
    """

    # Cluster ROIs based on PPC matrix, using sliding RT window
    # cluster_rois = cluster_rois_from_ppc_matrix(msdata.rois, ppc_matrix, peak_cor_rt_tol,
    #                                             min_ppc, min_overlap_ppc, min_cluster_size)

    # Cluster ROIs using Louvain algorithm
    cluster_rois = _perform_louvain_clustering(msdata, ppc_matrix,
                                               min_ppc=min_ppc, peak_cor_rt_tol=peak_cor_rt_tol,
                                               resolution=1.0, min_cluster_size=min_cluster_size,
                                               seed=123)

    # generate pseudo MS1 spectra
    pseudo_ms1_spectra = _map_cluster_labels_to_pseudo_ms1(msdata, cluster_rois)

    if save:
        save_pseudo_ms1_spectra(pseudo_ms1_spectra, msdata, save_dir)

    return pseudo_ms1_spectra


def cluster_rois_from_ppc_matrix(rois, ppc_matrix, peak_cor_rt_tol=0.05, min_ppc=0.8, min_overlap_ppc=0.95,
                                 min_cluster_size=6):
    """
    Cluster ROIs based on PPC matrix and RT tolerance
    """
    # Create a mapping of ROI IDs to their objects and sort them by RT
    roi_dict = {roi.id: roi for roi in rois}
    sorted_roi_ids = sorted(roi_dict.keys(), key=lambda id: roi_dict[id].rt)

    # Create a graph of ROI connections and collect ROIs with connections
    roi_graph = defaultdict(list)
    connected_rois = set()
    rows, cols = ppc_matrix.nonzero()
    for row, col in zip(rows, cols):
        if row < col and ppc_matrix[row, col] >= min_ppc:
            roi_a_id, roi_b_id = sorted_roi_ids[row], sorted_roi_ids[col]
            ppc = ppc_matrix[row, col]
            roi_graph[roi_a_id].append((roi_b_id, ppc))
            roi_graph[roi_b_id].append((roi_a_id, ppc))
            connected_rois.add(roi_a_id)
            connected_rois.add(roi_b_id)

    # Initial clustering based on sliding RT window
    initial_clusters = []
    window_start = 0
    while window_start < len(sorted_roi_ids):
        window_end = window_start
        start_rt = roi_dict[sorted_roi_ids[window_start]].rt
        current_cluster = []

        while window_end < len(sorted_roi_ids) and \
                roi_dict[sorted_roi_ids[window_end]].rt - start_rt <= 2 * peak_cor_rt_tol:
            if sorted_roi_ids[window_end] in connected_rois:
                current_cluster.append(sorted_roi_ids[window_end])
            window_end += 1

        if len(current_cluster) >= min_cluster_size:
            initial_clusters.append(current_cluster)

        # Move the window forward by peak_cor_rt_tol
        while window_start < len(sorted_roi_ids) and \
                roi_dict[sorted_roi_ids[window_start]].rt - start_rt <= peak_cor_rt_tol:
            window_start += 1

    # Find strongly connected components within initial clusters
    final_clusters = []
    for cluster in initial_clusters:
        components = find_connected_components(cluster, roi_graph)
        final_clusters.extend(comp for comp in components if len(comp) >= min_cluster_size)

    if final_clusters:
        print(f"Clustering summary (before overlap):")
        print(f"  Number of final clusters: {len(final_clusters)}")
        print(f"  Total ROIs in final clusters: {sum(len(cluster) for cluster in final_clusters)}")
        print(f"  Largest final cluster size: {max(len(cluster) for cluster in final_clusters)}")
        print(f"  Smallest final cluster size: {min(len(cluster) for cluster in final_clusters)}")
        print(
            f"  Average final cluster size: {sum(len(cluster) for cluster in final_clusters) / len(final_clusters):.2f}")
    #
    # # Allow overlaps for ROIs with strong connections to multiple clusters
    # for roi_id in set.union(*final_clusters):
    #     for cluster in final_clusters:
    #         if roi_id not in cluster:
    #             if any(neighbor[1] >= min_overlap_ppc for neighbor in roi_graph[roi_id] if neighbor[0] in cluster):
    #                 cluster.add(roi_id)
    #
    # if final_clusters:
    #     print(f"Final clustering summary (after overlap):")
    #     print(f"  Number of final clusters: {len(final_clusters)}")
    #     print(f"  Total ROIs in final clusters: {sum(len(cluster) for cluster in final_clusters)}")
    #     print(f"  Largest final cluster size: {max(len(cluster) for cluster in final_clusters)}")
    #     print(f"  Smallest final cluster size: {min(len(cluster) for cluster in final_clusters)}")
    #     print(
    #         f"  Average final cluster size: {sum(len(cluster) for cluster in final_clusters) / len(final_clusters):.2f}")

    # Convert to dictionary format
    cluster_rois = {i: cluster for i, cluster in enumerate(final_clusters)}

    return cluster_rois


def find_connected_components(roi_ids, roi_graph):
    """Find strongly connected components within a set of ROIs using depth-first search."""
    components = []
    visited = set()

    def dfs(roi_id, current_component):
        visited.add(roi_id)
        current_component.add(roi_id)
        for neighbor, _ in roi_graph[roi_id]:
            if neighbor in roi_ids and neighbor not in visited:
                dfs(neighbor, current_component)

    for roi_id in roi_ids:
        if roi_id not in visited:
            current_component = set()
            dfs(roi_id, current_component)
            components.append(current_component)

    return components


def _map_cluster_labels_to_pseudo_ms1(msdata, cluster_rois):
    """Map clustering labels to PseudoMS1 objects"""

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


def save_pseudo_ms1_spectra(pseudo_ms1_spectra, msdata, save_dir):
    if save_dir is None:
        path = os.path.join(msdata.params.single_file_dir, msdata.file_name + "_pseudoMS1.pkl")
    else:
        path = os.path.splitext(save_dir)[0] + "_pseudoMS1.pkl"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(pseudo_ms1_spectra, f)


def _refine_clusters_by_rt_window(cluster_rois, msdata, peak_cor_rt_tol, min_cluster_size):
    """
    Refine clusters by finding the optimal RT window that preserves the most data points within the given RT tolerance.

    :param cluster_rois: dictionary of cluster labels and ROI IDs
    :param msdata: MSData object containing ROIs
    :param peak_cor_rt_tol: maximum allowed RT difference within a cluster
    :return: refined dictionary of cluster labels and ROI IDs
    """
    refined_clusters = {}
    roi_dict = {roi.id: roi for roi in msdata.rois}

    for cluster_label, roi_ids in cluster_rois.items():

        # Sort ROIs by retention time
        sorted_rois = sorted(roi_ids, key=lambda roi_id: roi_dict[roi_id].rt)
        rts = [roi_dict[roi_id].rt for roi_id in sorted_rois]

        # Find the optimal RT window that preserves the most data points
        max_count = 0
        best_start_index = 0
        for start_index in range(len(rts)):
            end_index = start_index
            while end_index < len(rts) and rts[end_index] - rts[start_index] <= peak_cor_rt_tol:
                end_index += 1
            count = end_index - start_index
            if count > max_count:
                max_count = count
                best_start_index = start_index

        # Create the refined cluster with ROIs in the optimal RT window
        refined_cluster = sorted_rois[best_start_index:best_start_index + max_count]

        # If we have at least min_cluster_size ROIs in the refined cluster, add it to refined_clusters
        if len(refined_cluster) >= min_cluster_size:
            refined_clusters[cluster_label] = set(refined_cluster)

    return refined_clusters


def _perform_louvain_clustering(msdata, ppc_matrix,
                                min_ppc=0.8, peak_cor_rt_tol=0.05, min_cluster_size=6,
                                resolution=1.0, seed=123):
    """
    Cluster ROIs using Louvain algorithm based on high PPC scores
    :param msdata: MSData object containing ROIs
    :param ppc_matrix: sparse matrix of PPC scores
    :param min_ppc: min PPC score for clustering
    :param resolution: resolution parameter for Louvain clustering, lower values result in fewer clusters
    :param min_cluster_size: minimum number of ROIs in a cluster
    :param seed: random seed
    :return: dictionary of cluster labels and ROI IDs
    """
    # Create a graph with edges for high PPC scores only
    G = nx.Graph()
    rows, cols = ppc_matrix.nonzero()
    for row, col in zip(rows, cols):
        if row < col and ppc_matrix[row, col] >= min_ppc:
            G.add_edge(msdata.rois[row].id, msdata.rois[col].id, weight=ppc_matrix[row, col])

    # Perform Louvain clustering
    partition = nx.community.louvain_communities(G, weight='weight', resolution=resolution, seed=seed)

    # Convert partition to cluster dictionary and filter small clusters
    cluster_rois = {i: set(cluster) for i, cluster in enumerate(partition) if len(cluster) >= min_cluster_size}

    print(f"Louvain clustering summary:")
    print(f"  Number of clusters: {len(cluster_rois)}")
    print(f"  Total ROIs in clusters: {sum(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Largest cluster size: {max(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Smallest cluster size: {min(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Average cluster size: {sum(len(cluster) for cluster in cluster_rois.values()) / len(cluster_rois):.2f}")

    cluster_rois = _refine_clusters_by_rt_window(cluster_rois, msdata, peak_cor_rt_tol=peak_cor_rt_tol,
                                                 min_cluster_size=min_cluster_size)

    print(f"Final clustering summary:")
    print(f"  Number of clusters: {len(cluster_rois)}")
    print(f"  Total ROIs in clusters: {sum(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Largest cluster size: {max(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Smallest cluster size: {min(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Average cluster size: {sum(len(cluster) for cluster in cluster_rois.values()) / len(cluster_rois):.2f}")

    return cluster_rois
