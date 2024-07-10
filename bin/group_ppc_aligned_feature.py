"""
This is to get all features with high PPCs, generate pseudo MS1
"""
from collections import defaultdict

import numpy as np
import hdbscan
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def generate_pseudo_ms1(config, features, align_rt_tol=0.1, hdbscan_prob_cutoff=0.2):
    """
    Generate PseudoMS1 spectra
    :param config: Params
    :param features: list of Feature objects
    :param align_rt_tol: retention time tolerance for aligning features
    :param hdbscan_prob_cutoff: probability cutoff for HDBSCAN clustering
    :return: list of PseudoMS1 objects
    """
    feature_pair_ls = gen_ppc_for_aligned_features(config, features, rt_tol=align_rt_tol)

    all_cluster_labels, cluster_features = perform_hdbscan_for_feature_pairs(feature_pair_ls, hdbscan_prob_cutoff)

    pseudo_ms1_spectra = []

    # Create a dictionary to quickly look up features by their ID
    feature_dict = {feature.id: feature for feature in features}

    for cluster_label in all_cluster_labels:
        feature_ids = list(cluster_features[cluster_label])

        mz_ls = []
        rt_ls = []

        # Count detected features and sum intensities for each file
        file_counts = np.zeros(len(d.params.sample_names), dtype=int)
        file_intensities = np.zeros(len(d.params.sample_names))

        for feature_id in feature_ids:
            feature = feature_dict[feature_id]
            mz_ls.append(feature.mz)
            rt_ls.append(feature.rt)

            file_counts += feature.detected_seq
            file_intensities += feature.peak_height_seq

        # Select the file with the most detected features
        max_detected_file = np.argmax(file_counts)

        # If there's a tie, select the file with the highest total intensity
        if np.sum(file_counts == file_counts[max_detected_file]) > 1:
            max_intensity_files = np.where(file_counts == file_counts[max_detected_file])[0]
            max_detected_file = max_intensity_files[np.argmax(file_intensities[max_intensity_files])]

        # Get intensities from the selected file
        int_ls = [feature_dict[feature_id].peak_height_seq[max_detected_file] for feature_id in feature_ids]

        # Calculate average RT for the cluster
        avg_rt = sum(rt_ls) / len(rt_ls)

        # Create PseudoMS1 object
        pseudo_ms1 = PseudoMS1(mz_ls, int_ls, feature_ids, avg_rt)
        pseudo_ms1_spectra.append(pseudo_ms1)

    return pseudo_ms1_spectra


def perform_hdbscan_for_feature_pairs(feature_pairs, prob_cutoff=0.2):
    """
    perform HDBSCAN for a list of FeaturePair objects
    """

    # Get the unique feature IDs and map them to a contiguous range of indices
    unique_ids = list(set([fp.id_1 for fp in feature_pairs] + [fp.id_2 for fp in feature_pairs]))
    id_to_index = {id_: index for index, id_ in enumerate(unique_ids)}
    index_to_id = {index: id_ for id_, index in id_to_index.items()}
    num_features = len(unique_ids)

    # Create a similarity matrix initialized to zero
    similarity_matrix = np.zeros((num_features, num_features))

    # Fill in the similarity matrix with the average PPC scores
    for fp in feature_pairs:
        avg_ppc = np.mean(fp.ppc_seq)
        index_1 = id_to_index[fp.id_1]
        index_2 = id_to_index[fp.id_2]
        similarity_matrix[index_1, index_2] = avg_ppc
        similarity_matrix[index_2, index_1] = avg_ppc

    # Convert the similarity matrix to a distance matrix
    distance_matrix = 1 - similarity_matrix

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(metric='precomputed', prediction_data=True)
    clusterer.fit(distance_matrix)

    # Retrieve all cluster labels, excluding the noise cluster (-1)
    all_cluster_labels = set(label for label in clusterer.labels_ if label != -1)

    # Get soft cluster assignments
    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)

    # For each cluster, get feature IDs with probabilities exceeding prob_cutoff
    cluster_features = defaultdict(set)
    for index, probabilities in enumerate(soft_clusters):
        feature_id = index_to_id[index]
        # max_prob = max(probabilities)
        max_label = np.argmax(probabilities)

        assigned = False
        for label, probability in enumerate(probabilities):
            if label != -1 and probability >= prob_cutoff:  # Skip noise cluster
                cluster_features[label].add(feature_id)
                assigned = True

        # If the point wasn't assigned to any cluster, assign it to the highest probability cluster
        if not assigned and max_label != -1:
            cluster_features[max_label].add(feature_id)

    return all_cluster_labels, cluster_features


def gen_ppc_for_aligned_features(config, features, rt_tol=0.1):
    """
    generate ppc scores for aligned features
    :param config: Params
    :param features: list of Feature objects
    :param rt_tol: rt tolerance for aligning features
    :return: list of FeaturePair objects
    """

    # all features sorted by id
    features.sort(key=lambda x: x.id)

    # Generate feature pairs
    feature_pairs = []
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features[i + 1:], start=i + 1):
            if abs(f1.rt - f2.rt) <= rt_tol:
                fp = FeaturePair(f1.id, f2.id, f1.roi_id_seq, f2.roi_id_seq)
                feature_pairs.append(fp)

    # Create a dictionary for quick access to feature pairs
    fp_dict = {fp.id: fp for fp in feature_pairs}

    # Set up multiprocessing
    pool = mp.Pool(processes=mp.cpu_count())
    path = os.path.join(config.single_file_dir)

    # Partial function for multiprocessing
    process_file = partial(process_single_file, path=path, feature_pairs=feature_pairs)

    # Process files in parallel
    results = list(tqdm(pool.imap(process_file, enumerate(config.sample_names)),
                        total=len(config.sample_names),
                        desc="Processing files"))

    # Close the pool
    pool.close()
    pool.join()

    # Update feature pairs with results
    for file_results in results:
        if len(file_results) > 0:
            for fp_id, i, this_ppc in file_results:
                fp_dict[fp_id].ppc_seq[i] = this_ppc

    return list(fp_dict.values())


def process_single_file(file_name, path, feature_pairs, i):
    """
    read a single ppc file and add to feature_pairs
    """
    file = file_name + '_peakCor.npy'
    file_path = os.path.join(path, file)
    file_ppc = np.load(file_path)

    if len(file_ppc) == 0:
        return []

    results = []
    for fp in feature_pairs:
        this_ppc = _get_ppc_from_single_file(file_ppc, fp.file_roi_id_seq_1, fp.file_roi_id_seq_2)
        results.append((fp.id, i, this_ppc))

    return results


def _get_ppc_from_single_file(file_ppc_arr, roi_id1, roi_id2):
    """
    return the ppc score given two roi ids within a single file
    """
    if roi_id1 > roi_id2:
        roi_id1, roi_id2 = roi_id2, roi_id1

    # Find the row with the given ROI IDs
    matches = (file_ppc_arr[:, 0] == roi_id1) & (file_ppc_arr[:, 1] == roi_id2)
    match_indices = np.where(matches)[0]

    if match_indices.size > 0:
        return file_ppc_arr[match_indices[0], 2]
    else:
        return 0.0


class FeaturePair:
    """
    A class to model a pair of aligned features
    """
    # aligned id (smaller), aligned id (larger), single file id array, single file id array, ppc array
    def __init__(self, id_1, id_2, arr_1, arr_2):

        self.id_1 = min(id_1, id_2)
        self.id_2 = max(id_1, id_2)
        self.file_roi_id_seq_1 = arr_1      # ROI ID from individual files (-1 if not detected or gap filled)
        self.file_roi_id_seq_2 = arr_2      # ROI ID from individual files (-1 if not detected or gap filled)
        self.ppc_seq = np.zeros(len(arr_1))           # PPC score array


class PseudoMS1:
    """
    A class to model pseudo MS1 spectrum
    """
    def __init__(self, mz_ls, int_ls, feature_ids, rt):

        self.mzs = mz_ls  # list
        self.intensities = int_ls  # list
        self.feature_ids = feature_ids  # list of feature ids in aligned feature table
        self.rt = rt

        # annotation
        self.annotated = False
        self.annotation_ls = []  # list of SpecAnnotation objects


class SpecAnnotation:
    """
    A class to model spectral annotation
    """

    def __init__(self, score, matched_peak):
        self.score = score
        self.matched_peak = matched_peak

        # database info
        self.db_id = None
        self.name = None
        self.precursor_type = None
        self.formula = None
        self.inchikey = None
        self.instrument_type = None
        self.collision_energy = None
