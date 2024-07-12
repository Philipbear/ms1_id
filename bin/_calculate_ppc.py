"""
Note: This script requires Numba to be installed. You can install it using:
pip install numba

The first run of this script might be slower due to Numba's compilation time,
but subsequent runs should be much faster.
"""

import numpy as np
import os
from scipy.sparse import lil_matrix, save_npz, load_npz
from numba import njit, prange


@njit(parallel=True)
def calc_ppc_matrix(roi_ids, roi_rts, roi_scan_idx_seqs, roi_int_seqs, rt_tol):
    n_rois = len(roi_ids)
    ppc_values = []
    ppc_rows = []
    ppc_cols = []

    for i in prange(n_rois):
        for j in range(i + 1, n_rois):
            if abs(roi_rts[i] - roi_rts[j]) <= rt_tol:
                ppc = peak_peak_correlation_numba(
                    roi_scan_idx_seqs[i], roi_int_seqs[i],
                    roi_scan_idx_seqs[j], roi_int_seqs[j]
                )
                ppc_values.append(ppc)
                ppc_rows.append(roi_ids[i])
                ppc_cols.append(roi_ids[j])

    return ppc_values, ppc_rows, ppc_cols


@njit
def peak_peak_correlation_numba(scan_idx1, int_seq1, scan_idx2, int_seq2):
    common_scans = intersect1d_numba(scan_idx1, scan_idx2)

    if len(common_scans) < 5:
        return 1.0

    int1 = extract_common_intensities(scan_idx1, int_seq1, common_scans)
    int2 = extract_common_intensities(scan_idx2, int_seq2, common_scans)

    if np.all(int1 == int1[0]) or np.all(int2 == int2[0]):
        return 1.0

    mean1, mean2 = np.mean(int1), np.mean(int2)
    diff1, diff2 = int1 - mean1, int2 - mean2
    numerator = np.sum(diff1 * diff2)
    denominator = np.sqrt(np.sum(diff1 ** 2) * np.sum(diff2 ** 2))

    return numerator / denominator if denominator != 0 else 1.0


@njit
def intersect1d_numba(arr1, arr2):
    arr1 = np.sort(arr1)
    arr2 = np.sort(arr2)
    i, j = 0, 0
    result = []
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            i += 1
        elif arr1[i] > arr2[j]:
            j += 1
        else:
            result.append(arr1[i])
            i += 1
            j += 1
    return np.array(result)


@njit
def extract_common_intensities(scan_idx, int_seq, common_scans):
    result = []
    for scan in common_scans:
        idx = np.searchsorted(scan_idx, scan)
        if idx < len(scan_idx) and scan_idx[idx] == scan:
            result.append(int_seq[idx])
    return np.array(result)


def calc_all_ppc(d, rt_tol=0.1, save=True, path=None):
    """
    A function to calculate all peak-peak correlations and store them efficiently
    :param d: MSData object
    :param rt_tol: retention time tolerance for aligning features
    :param save: whether to save the results
    :param path: path to save the results
    """
    rois = sorted(d.rois, key=lambda x: x.id)
    n_rois = len(rois)

    roi_ids = np.array([roi.id for roi in rois])
    roi_rts = np.array([roi.rt for roi in rois])
    roi_scan_idx_seqs = [np.array(roi.scan_idx_seq) for roi in rois]
    roi_int_seqs = [np.array(roi.int_seq) for roi in rois]

    ppc_values, ppc_rows, ppc_cols = calc_ppc_matrix(
        roi_ids, roi_rts, roi_scan_idx_seqs, roi_int_seqs, rt_tol
    )

    ppc_matrix = lil_matrix((n_rois, n_rois))
    for value, row, col in zip(ppc_values, ppc_rows, ppc_cols):
        ppc_matrix[row, col] = value
        ppc_matrix[col, row] = value  # Symmetric matrix

    if save and path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_npz(path, ppc_matrix)

    return ppc_matrix.tocsr()
