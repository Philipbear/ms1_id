"""
This is to get all features with high PPCs, generate pseudo MS1
"""
from collections import defaultdict
import numpy as np
from numba import njit
from hdbscan import HDBSCAN, all_points_membership_vectors
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz

from _utils import PseudoMS1, FeaturePair


def generate_pseudo_ms1(config, features, peak_cor_rt_tol=0.1, hdbscan_prob_cutoff=0.2,
                        file_selection_mode='sum_normalized'):
    """
    Generate PseudoMS1 spectra
    :param config: Params
    :param features: list of Feature objects
    :param peak_cor_rt_tol: retention time tolerance for aligning features
    :param hdbscan_prob_cutoff: probability cutoff for HDBSCAN clustering
    :param file_selection_mode: 'max_detection' or 'sum_normalized'
    :return: list of PseudoMS1 objects
    """
    feature_pair_ls = _gen_ppc_for_aligned_features(config, features, rt_tol=peak_cor_rt_tol)

    all_cluster_labels, cluster_features = _perform_hdbscan_for_feature_pairs(feature_pair_ls, hdbscan_prob_cutoff)

    pseudo_ms1_spectra = []
    feature_dict = {feature.id: feature for feature in features}

    for cluster_label in all_cluster_labels:
        feature_ids = list(cluster_features[cluster_label])
        mz_ls, rt_ls = [], []
        file_counts = np.zeros(len(config.sample_names), dtype=int)
        file_intensities = np.zeros(len(config.sample_names))

        for feature_id in feature_ids:
            feature = feature_dict[feature_id]
            mz_ls.append(feature.mz)
            rt_ls.append(feature.rt)
            file_counts += feature.detected_seq
            file_intensities += feature.peak_height_seq

        if file_selection_mode == 'max_detection':
            max_detected_file = np.argmax(file_counts)
            if np.sum(file_counts == file_counts[max_detected_file]) > 1:
                max_intensity_files = np.where(file_counts == file_counts[max_detected_file])[0]
                max_detected_file = max_intensity_files[np.argmax(file_intensities[max_intensity_files])]
            int_ls = [feature_dict[feature_id].peak_height_seq[max_detected_file] for feature_id in feature_ids]

        elif file_selection_mode == 'sum_normalized':

            # Normalize intensities for each file across features
            normalized_intensities = []

            n_features = len(feature_ids)
            n_files = len(config.sample_names)
            min_detection_threshold = 0.2 * n_features  # 20% of total features
            replacement_value = config.int_tol * 0.1

            files_passed_threshold = 0

            for file_index in range(n_files):
                file_intensities = []
                detected_features = 0

                # Count detected features and collect intensities for this file
                for feature_id in feature_ids:
                    feature = feature_dict[feature_id]
                    intensity = feature.peak_height_seq[file_index]

                    if intensity > 0:
                        detected_features += 1

                    file_intensities.append(intensity)

                # Check if at least 20% of features are detected in this file
                if detected_features >= min_detection_threshold:
                    files_passed_threshold += 1

                    # Replace zero intensities with config.int_tol * 0.1
                    file_intensities = np.array(file_intensities)
                    file_intensities[file_intensities == 0] = replacement_value

                    # Normalize
                    total_intensity = np.sum(file_intensities)
                    normalized = file_intensities / total_intensity
                    normalized_intensities.append(normalized)

                else:
                    # If not enough features are detected, use zeros
                    normalized_intensities.append(np.zeros(n_features))

            # Check if any files passed the threshold
            if files_passed_threshold == 0:
                # If no files passed, use the original 'max_detection' method
                max_detected_file = np.argmax(file_counts)
                if np.sum(file_counts == file_counts[max_detected_file]) > 1:
                    max_intensity_files = np.where(file_counts == file_counts[max_detected_file])[0]
                    max_detected_file = max_intensity_files[np.argmax(file_intensities[max_intensity_files])]
                int_ls = [feature_dict[feature_id].peak_height_seq[max_detected_file] for feature_id in feature_ids]
            else:
                # Convert to numpy array for easier manipulation
                normalized_intensities = np.array(normalized_intensities)
                # Sum normalized intensities across files
                summed_normalized_intensities = np.sum(normalized_intensities, axis=0)
                # Calculate the final intensity for each feature
                int_ls = []
                for i, feature_id in enumerate(feature_ids):
                    feature = feature_dict[feature_id]
                    # Use the maximum intensity of the feature across all files
                    max_intensity = np.max(feature.peak_height_seq)
                    # Multiply by the summed normalized intensity
                    int_ls.append(max_intensity * summed_normalized_intensities[i])

        else:
            raise ValueError("Invalid file_selection_mode. Choose 'max_detection' or 'sum_normalized'.")

        avg_rt = sum(rt_ls) / len(rt_ls)

        pseudo_ms1 = PseudoMS1(mz_ls, int_ls, feature_ids, avg_rt)
        pseudo_ms1_spectra.append(pseudo_ms1)

    return pseudo_ms1_spectra


def _perform_hdbscan_for_feature_pairs(feature_pairs, prob_cutoff=0.2, min_cluster_size=5, min_samples=None):
    """
    Perform HDBSCAN clustering on FeaturePair objects
    :param feature_pairs: list of FeaturePair objects
    :param prob_cutoff: probability cutoff for HDBSCAN clustering
    :param min_cluster_size: minimum cluster size for HDBSCAN
    :param min_samples: minimum number of samples for HDBSCAN
    :return: all cluster labels, cluster features
    """
    unique_ids = list(set([fp.id_1 for fp in feature_pairs] + [fp.id_2 for fp in feature_pairs]))
    id_to_index = {id_: index for index, id_ in enumerate(unique_ids)}
    index_to_id = {index: id_ for id_, index in id_to_index.items()}
    num_features = len(unique_ids)

    similarity_matrix = np.zeros((num_features, num_features))
    for fp in feature_pairs:
        avg_ppc = np.mean(fp.ppc_seq)
        index_1, index_2 = id_to_index[fp.id_1], id_to_index[fp.id_2]
        similarity_matrix[index_1, index_2] = similarity_matrix[index_2, index_1] = avg_ppc

    distance_matrix = 1 - similarity_matrix

    # Perform HDBSCAN clustering
    clusterer = HDBSCAN(metric='precomputed',
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        prediction_data=True)  # Enable prediction data for soft clustering
    clusterer.fit(distance_matrix)

    # Get soft clustering probabilities
    soft_clusters = all_points_membership_vectors(clusterer)

    cluster_features = defaultdict(set)
    all_cluster_labels = set()

    for index, probabilities in enumerate(soft_clusters):
        feature_id = index_to_id[index]

        # Assign to all clusters where probability exceeds the cutoff
        for cluster, prob in enumerate(probabilities):
            if prob >= prob_cutoff:
                cluster_features[cluster].add(feature_id)
                all_cluster_labels.add(cluster)

        # If not assigned to any cluster, assign to the cluster with highest probability
        if all(prob < prob_cutoff for prob in probabilities):
            best_cluster = np.argmax(probabilities)
            cluster_features[best_cluster].add(feature_id)
            all_cluster_labels.add(best_cluster)

    # Remove noise cluster (-1) if present
    if -1 in all_cluster_labels:
        all_cluster_labels.remove(-1)
        del cluster_features[-1]

    return all_cluster_labels, cluster_features


def _gen_ppc_for_aligned_features(config, features, rt_tol=0.1):
    """
    Generate peak-peak correlations for aligned features
    """
    features.sort(key=lambda x: x.id)

    # Extract ids, rts, and roi_id_seqs separately
    ids = np.array([f.id for f in features])
    rts = np.array([f.rt for f in features])
    roi_id_seqs = [f.roi_id_seq for f in features]

    # Use Numba-optimized function to find valid pairs
    valid_pairs = find_valid_pairs(ids, rts, rt_tol)

    # Create FeaturePair objects
    feature_pairs = [
        FeaturePair(
            ids[i], ids[j],
            roi_id_seqs[i], roi_id_seqs[j]
        )
        for i, j in valid_pairs
    ]

    fp_dict = {fp.id: fp for fp in feature_pairs}
    path = os.path.join(config.single_file_dir)

    for i, file_name in tqdm(enumerate(config.sample_names), total=len(config.sample_names),
                             desc="Getting PPC scores"):
        ppc_matrix = load_ppc_matrix(os.path.join(path, file_name + '_peakCor.npz'))
        for fp in feature_pairs:
            roi_id1 = fp.file_roi_id_seq_1[i]
            roi_id2 = fp.file_roi_id_seq_2[i]
            if roi_id1 != -1 and roi_id2 != -1:
                fp_dict[fp.id].ppc_seq[i] = get_ppc_score(ppc_matrix, roi_id1, roi_id2)

    return list(fp_dict.values())


@njit
def find_valid_pairs(ids, rts, rt_tol):
    n = len(ids)
    valid_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(rts[i] - rts[j]) <= rt_tol:
                valid_pairs.append((i, j))
    return valid_pairs


def load_ppc_matrix(file_path):
    """
    Load the PPC matrix from a file
    """
    return load_npz(file_path)


def get_ppc_score(ppc_matrix, roi_id1, roi_id2):
    """
    Get the PPC score for a pair of ROIs
    :param ppc_matrix: sparse matrix of PPC scores
    :param roi_id1: ID of the first ROI
    :param roi_id2: ID of the second ROI
    :return: PPC score
    """
    return ppc_matrix[roi_id1, roi_id2]
