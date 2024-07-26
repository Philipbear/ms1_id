"""
This is to get all features with high PPCs, generate pseudo MS1 for a single file
"""
import os
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np

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


def generate_pseudo_ms1(msdata, ppc_matrix,
                        peak_cor_rt_tol=0.05, min_ppc=0.8, roi_min_length=3,
                        min_cluster_size=6, resolution=0.05,
                        save=False, save_dir=None):
    """
    Generate pseudo MS1 spectra for a single file
    :param msdata: MSData object
    :param ppc_matrix: sparse matrix of PPC scores
    :param roi_min_length: minimum length of ROIs to consider for clustering
    :param peak_cor_rt_tol: RT tolerance for clustering
    :param min_ppc: minimum PPC score for clustering
    :param min_cluster_size: minimum number of ROIs in a cluster
    :param resolution: resolution parameter for Louvain clustering
    :param save: whether to save the pseudo MS1 spectra
    :param save_dir: directory to save the pseudo MS1 spectra
    """

    # Cluster ROIs using Louvain algorithm
    # cluster_rois = _perform_louvain_clustering(msdata, ppc_matrix, roi_min_length=roi_min_length,
    #                                            min_ppc=min_ppc, peak_group_rt_tol=peak_group_rt_tol,
    #                                            resolution=resolution, min_cluster_size=min_cluster_size,
    #                                            seed=123)
    # plot_louvain_clustering_network(msdata, cluster_rois, ppc_matrix)
    # pseudo_ms1_spectra = _map_cluster_labels_to_pseudo_ms1(msdata, cluster_rois)

    pseudo_ms1_spectra = _perform_clustering(msdata, ppc_matrix, min_ppc=min_ppc, peak_cor_rt_tol=peak_cor_rt_tol,
                                             min_cluster_size=min_cluster_size, roi_min_length=roi_min_length)

    plot_mz_rt_scatter_with_pseudo_ms1(msdata, pseudo_ms1_spectra, roi_min_length=roi_min_length)

    if save:
        save_pseudo_ms1_spectra(pseudo_ms1_spectra, msdata, save_dir)

    return pseudo_ms1_spectra


def _perform_clustering(msdata, ppc_matrix, min_ppc=0.8, peak_cor_rt_tol=0.05,
                        min_cluster_size=6, roi_min_length=3):
    """
    Perform clustering on ROIs based on PPC scores and m/z values.

    :param msdata: MSData object containing ROIs
    :param ppc_matrix: scipy.sparse.csr_matrix of PPC scores
    :param min_ppc: Minimum PPC score to consider for clustering
    :param peak_cor_rt_tol: RT tolerance for clustering
    :param min_cluster_size: Minimum number of ROIs in a cluster
    :param roi_min_length: Minimum length of ROIs to consider for clustering
    :return: List of PseudoMS1 objects
    """
    # Filter ROIs based on minimum length and isotope status
    valid_rois = [roi for roi in msdata.rois if roi.length >= roi_min_length and not roi.is_isotope]

    # Sort ROIs by m/z values
    sorted_rois = sorted(valid_rois, key=lambda roi: roi.mz)

    # Create a new PPC matrix with only valid ROIs
    valid_indices = [msdata.rois.index(roi) for roi in valid_rois]
    new_ppc_matrix = ppc_matrix[valid_indices][:, valid_indices]

    pseudo_ms1_spectra = []

    for i, roi in enumerate(sorted_rois):
        # Find all ROIs with PPC scores above the threshold
        cluster_indices = new_ppc_matrix[i].nonzero()[1]
        cluster_scores = new_ppc_matrix[i, cluster_indices].toarray().flatten()
        cluster_indices = cluster_indices[cluster_scores >= min_ppc]

        if len(cluster_indices) >= min_cluster_size:
            # Form a pseudo MS1 spectrum
            cluster_rois = [sorted_rois[idx] for idx in cluster_indices]
            mz_ls = [roi.mz for roi in cluster_rois]
            int_ls = [roi.peak_height for roi in cluster_rois]
            roi_ids = [roi.id for roi in cluster_rois]
            avg_rt = np.mean([roi.rt for roi in cluster_rois])

            pseudo_ms1 = PseudoMS1(mz_ls, int_ls, roi_ids, msdata.file_name, avg_rt)
            pseudo_ms1_spectra.append(pseudo_ms1)

    # Sort pseudo MS1 spectra by RT
    pseudo_ms1_spectra.sort(key=lambda x: x.rt)

    # First pass: store subset information
    subset_info = {i: set() for i in range(len(pseudo_ms1_spectra))}
    for i, spec1 in enumerate(pseudo_ms1_spectra):
        set1 = set(spec1.roi_ids)
        for j, spec2 in enumerate(pseudo_ms1_spectra[i + 1:], start=i + 1):
            if abs(spec1.rt - spec2.rt) > peak_cor_rt_tol:
                break

            set2 = set(spec2.roi_ids)

            if set1 == set2:
                subset_info[i].add(j)
                subset_info[j].add(i)
            elif set1.issubset(set2):
                subset_info[i].add(j)
            elif set2.issubset(set1):
                subset_info[j].add(i)

    # Second pass: determine spectra to keep
    spectra_to_keep = set(range(len(pseudo_ms1_spectra)))
    for i in range(len(pseudo_ms1_spectra)):
        if i in spectra_to_keep:
            # Remove all subsets of this spectrum
            spectra_to_keep -= subset_info[i]

    # Create the final list of non-redundant spectra
    non_redundant_spectra = [pseudo_ms1_spectra[i] for i in sorted(spectra_to_keep)]

    return non_redundant_spectra


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

'''
def _refine_clusters_by_rt_window(cluster_rois, msdata, peak_group_rt_tol, min_cluster_size):
    """
    Refine clusters by finding the optimal RT window that preserves the most data points within the given RT tolerance.

    :param cluster_rois: dictionary of cluster labels and ROI IDs
    :param msdata: MSData object containing ROIs
    :param peak_group_rt_tol: maximum allowed RT difference within a cluster
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
            while end_index < len(rts) and rts[end_index] - rts[start_index] <= peak_group_rt_tol:
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


def _perform_louvain_clustering(msdata, ppc_matrix, roi_min_length=3,
                                min_ppc=0.8, peak_group_rt_tol=0.05, min_cluster_size=6,
                                resolution=0.5, seed=123):
    """
    Cluster ROIs using Louvain algorithm based on high PPC scores
    :param msdata: MSData object containing ROIs
    :param ppc_matrix: sparse matrix of PPC scores
    :param roi_min_length: minimum length of ROIs to consider for clustering
    :param min_ppc: min PPC score for clustering
    :param peak_group_rt_tol: maximum allowed RT difference within a cluster
    :param resolution: resolution parameter for Louvain clustering, lower values result in fewer clusters
    :param min_cluster_size: minimum number of ROIs in a cluster
    :param seed: random seed
    :return: dictionary of cluster labels and ROI IDs
    """
    # Create a graph with edges for high PPC scores only, checking ROI length on-the-fly
    G = nx.Graph()
    rows, cols = ppc_matrix.nonzero()
    for row, col in zip(rows, cols):
        if (row < col and ppc_matrix[row, col] >= min_ppc and msdata.rois[row].length >= roi_min_length
                and msdata.rois[col].length >= roi_min_length and not msdata.rois[row].is_isotope and
                not msdata.rois[col].is_isotope):
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

    cluster_rois = _refine_clusters_by_rt_window(cluster_rois, msdata, peak_group_rt_tol=peak_group_rt_tol,
                                                 min_cluster_size=min_cluster_size)

    print(f"Final clustering summary:")
    print(f"  Number of clusters: {len(cluster_rois)}")
    print(f"  Total ROIs in clusters: {sum(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Largest cluster size: {max(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Smallest cluster size: {min(len(cluster) for cluster in cluster_rois.values())}")
    print(f"  Average cluster size: {sum(len(cluster) for cluster in cluster_rois.values()) / len(cluster_rois):.2f}")

    return cluster_rois


def plot_louvain_clustering_network(msdata, cluster_rois, ppc_matrix, min_ppc=0.8, max_nodes=100):
    """
    Plot Louvain clustering network results.

    :param msdata: MSData object containing ROIs
    :param cluster_rois: Dictionary of cluster labels and ROI IDs
    :param ppc_matrix: Sparse matrix of PPC scores
    :param min_ppc: Minimum PPC score to include in the plot
    :param max_nodes: Maximum number of nodes to plot (to avoid overcrowding)
    """
    G = nx.Graph()

    # Create a dictionary mapping ROI IDs to their m/z values
    roi_mz_dict = {roi.id: roi.mz for roi in msdata.rois}

    # Add nodes and edges
    for cluster, roi_ids in cluster_rois.items():
        for roi_id in roi_ids:
            G.add_node(roi_id, mz=roi_mz_dict[roi_id])

    rows, cols = ppc_matrix.nonzero()
    for row, col in zip(rows, cols):
        if row < col and ppc_matrix[row, col] >= min_ppc:
            roi_a_id, roi_b_id = msdata.rois[row].id, msdata.rois[col].id
            G.add_edge(roi_a_id, roi_b_id, weight=ppc_matrix[row, col])

    # Limit the number of nodes to avoid overcrowding
    if len(G) > max_nodes:
        G = nx.Graph(G.subgraph(list(G.nodes)[:max_nodes]))

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', ax=ax)

    # Draw edges with color based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Create a lighter color map
    cmap = plt.cm.get_cmap('Blues')
    lighter_cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", cmap(0.7)])
    norm = mcolors.Normalize(vmin=min_ppc, vmax=1.0)

    nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=lighter_cmap, edge_vmin=min_ppc, edge_vmax=1.0, ax=ax,
                           width=0.5)

    # Add m/z labels to nodes
    nx.draw_networkx_labels(G, pos, {node: f"{G.nodes[node]['mz']:.2f}" for node in G.nodes()}, font_size=5.5, ax=ax)

    ax.set_title("Louvain Clustering Network")
    ax.axis('off')

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('PPC Score')

    plt.tight_layout()
    plt.show()
'''


def plot_mz_rt_scatter_with_pseudo_ms1(msdata, pseudo_ms1_spectra, roi_min_length=3):
    """
    Plot RT-m/z scatter plot of all ROIs and lines parallel to RT axis for pseudo MS1 spectra.

    :param msdata: MSData object containing ROIs
    :param pseudo_ms1_spectra: List of PseudoMS1 objects
    :param roi_min_length: Minimum length of ROIs to plot
    """
    plt.figure(figsize=(12, 8))

    # Plot all ROIs
    rt_values = [roi.rt for roi in msdata.rois if roi.length >= roi_min_length and not roi.is_isotope]
    mz_values = [roi.mz for roi in msdata.rois if roi.length >= roi_min_length and not roi.is_isotope]
    plt.scatter(rt_values, mz_values, alpha=0.5, s=5, label='Metabolic features')

    # Plot lines parallel to RT axis for pseudo MS1 spectra
    for pseudo_ms1 in pseudo_ms1_spectra:
        rt = pseudo_ms1.rt
        mz_min, mz_max = min(pseudo_ms1.mzs), max(pseudo_ms1.mzs)
        plt.vlines(x=rt, ymin=mz_min, ymax=mz_max, color='r', alpha=0.3, linestyle='--')

    plt.xlabel('Retention Time (min)')
    plt.ylabel('m/z')

    plt.title('Metabolic feature scatter plot with pseudo MS1 spectra')
    plt.legend()
    plt.tight_layout()
    plt.show()
