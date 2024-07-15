"""
Note: This script requires Numba to be installed. You can install it using:
pip install numba

The first run of this script might be slower due to Numba's compilation time,
but subsequent runs should be much faster.
"""

import os

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, save_npz


@njit
def peak_peak_correlation_numba(scan_idx1, int_seq1, scan_idx2, int_seq2):
    i, j = 0, 0
    sum_x, sum_y, sum_xy, sum_x2, sum_y2 = 0.0, 0.0, 0.0, 0.0, 0.0
    n = 0

    while i < len(scan_idx1) and j < len(scan_idx2):
        if scan_idx1[i] < scan_idx2[j]:
            i += 1
        elif scan_idx1[i] > scan_idx2[j]:
            j += 1
        else:
            x, y = int_seq1[i], int_seq2[j]
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y
            n += 1
            i += 1
            j += 1

    if n < 3:
        return 0.0

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

    return numerator / denominator if denominator != 0 else 1.0


@njit
def calc_ppc_matrix_numba(roi_ids, roi_rts, roi_scan_idx_seqs, roi_int_seqs, roi_lengths, rt_tol):
    n_rois = len(roi_ids)
    rows = np.zeros(n_rois * n_rois, dtype=np.int32)
    cols = np.zeros(n_rois * n_rois, dtype=np.int32)
    data = np.zeros(n_rois * n_rois, dtype=np.float64)  # Change to float64
    counter = 0

    for i in range(n_rois):
        start_i = np.sum(roi_lengths[:i])
        end_i = start_i + roi_lengths[i]
        for j in range(i + 1, n_rois):
            if abs(roi_rts[i] - roi_rts[j]) <= rt_tol:
                start_j = np.sum(roi_lengths[:j])
                end_j = start_j + roi_lengths[j]
                ppc = peak_peak_correlation_numba(
                    roi_scan_idx_seqs[start_i:end_i], roi_int_seqs[start_i:end_i],
                    roi_scan_idx_seqs[start_j:end_j], roi_int_seqs[start_j:end_j]
                )
                rows[counter] = i
                cols[counter] = j
                data[counter] = ppc
                counter += 1

    return rows[:counter], cols[:counter], data[:counter]


def calc_all_ppc(d, rt_tol=0.1, save=True, path=None):
    """
    Calculate peak-peak correlation matrix for all ROIs in the dataset
    """

    rois = sorted(d.rois, key=lambda x: x.id)
    n_rois = len(rois)

    roi_ids = np.array([roi.id for roi in rois], dtype=np.int32)
    roi_rts = np.array([roi.rt for roi in rois], dtype=np.float64)  # Change to float64

    roi_lengths = np.array([len(roi.scan_idx_seq) for roi in rois], dtype=np.int32)
    roi_scan_idx_seqs = np.concatenate([np.array(roi.scan_idx_seq, dtype=np.int32) for roi in rois])
    roi_int_seqs = np.concatenate([np.array(roi.int_seq, dtype=np.float64) for roi in rois])  # Change to float64

    rows, cols, data = calc_ppc_matrix_numba(
        roi_ids, roi_rts, roi_scan_idx_seqs, roi_int_seqs, roi_lengths, rt_tol
    )

    ppc_matrix = csr_matrix((data, (rows, cols)), shape=(n_rois, n_rois), dtype=np.float64)  # Specify dtype
    ppc_matrix = ppc_matrix + ppc_matrix.T
    ppc_matrix.setdiag(1.0)

    if save and path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_npz(path, ppc_matrix)

    return ppc_matrix
