"""
This is to get all features with high PPCs, generate pseudo MS1 for a single file
"""
import os
import pickle
import numpy as np

from ms1_id.lcms.utils import PseudoMS2
from ms1_id.lcms.export import write_pseudoms2_to_mgf


def retrieve_pseudo_ms2_spectra(config):
    """
    Retrieve pseudo MS1 spectra for all files
    :param config: Config object
    :return: dictionary of pseudo MS2 spectra
    """
    pseudo_ms2_spectra = []

    files = os.listdir(config.single_file_dir)
    for file in files:
        if file.endswith('_pseudoMS2_annotated.pkl'):
            try:
                with open(os.path.join(config.single_file_dir, file), 'rb') as f:
                    new_pseudo_ms2_spectra = pickle.load(f)
                    pseudo_ms2_spectra.extend(new_pseudo_ms2_spectra)
            except:
                continue

    return pseudo_ms2_spectra


def generate_pseudo_ms2(msdata, ppc_matrix,
                        mz_tol=0.01,
                        min_ppc=0.8,
                        min_cluster_size=4,
                        roi_min_length=3,
                        roi_max_rt_range=2.0,
                        save_dir=None):
    """
    Generate pseudo MS2 spectra for a single file
    :param msdata: MSData object
    :param ppc_matrix: sparse matrix of PPC scores
    :param roi_min_length: minimum length of ROIs to consider for clustering
    :param roi_max_rt_range: maximum RT range of ROIs to consider for clustering
    :param mz_tol: m/z tolerance
    :param min_ppc: minimum PPC score for clustering
    :param min_cluster_size: minimum number of ROIs in a cluster
    :param save_dir: directory to save the pseudo MS2 spectra
    """

    pseudo_ms2_spectra = _perform_clustering(msdata, ppc_matrix,
                                             mz_tol=mz_tol,
                                             min_ppc=min_ppc,
                                             min_cluster_size=min_cluster_size,
                                             roi_min_length=roi_min_length,
                                             roi_max_rt_range=roi_max_rt_range)

    write_pseudoms2_to_mgf(pseudo_ms2_spectra, save_dir, msdata.file_name)

    return pseudo_ms2_spectra


def _perform_clustering(msdata, ppc_matrix, mz_tol=0.01, min_ppc=0.8, min_cluster_size=4,
                        roi_min_length=3, roi_max_rt_range=1.0):
    """
    Perform clustering on ROIs based on PPC scores and m/z values,
    considering only m/z values smaller than the target m/z for each ROI.
    """
    # Filter ROIs based on minimum length and isotope status using a generator expression
    valid_rois = (roi for roi in msdata.rois if roi.length >= roi_min_length
                  # and (max(roi.rt_seq) - min(roi.rt_seq) <= roi_max_rt_range)
                  and not roi.is_isotope)

    # Sort ROIs by m/z values and create a mapping
    sorted_rois = sorted(valid_rois, key=lambda roi: roi.mz)

    # Create a new PPC matrix with only valid ROIs
    valid_indices = [msdata.rois.index(roi) for roi in sorted_rois]
    new_ppc_matrix = ppc_matrix[valid_indices][:, valid_indices]

    pseudo_ms2_spectra = []

    for i, roi in enumerate(sorted_rois):
        t_mz = roi.mz  # Set the target m/z as the current ROI's m/z

        # Find all ROIs with PPC scores above the threshold
        cluster_indices = new_ppc_matrix[i].nonzero()[1]
        cluster_scores = new_ppc_matrix[i, cluster_indices].toarray().flatten()

        # cluster_indices = cluster_indices[(cluster_scores >= min_ppc) &
        #                                   (np.array([sorted_rois[idx].mz for idx in cluster_indices]) <= t_mz + 1e-2)]
        cluster_indices = cluster_indices[cluster_scores >= min_ppc]

        if len(cluster_indices) >= min_cluster_size:
            # Form a pseudo MS2 spectrum
            cluster_rois = [sorted_rois[idx] for idx in cluster_indices]

            mz_ls = [roi.mz for roi in cluster_rois]
            int_ls = [roi.peak_height for roi in cluster_rois]
            roi_ids = [roi.id for roi in cluster_rois]
            # rt = np.mean([roi.rt for roi in cluster_rois])
            rt = roi.rt

            pseudo_ms2_spectra.append((t_mz, mz_ls, int_ls, roi_ids, msdata.file_name, rt, mz_tol))

    # Convert to PseudoMS2 objects after all processing
    pseudo_ms2_spectra = [PseudoMS2(*spec) for spec in pseudo_ms2_spectra]

    return pseudo_ms2_spectra

