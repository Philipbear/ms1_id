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
from scipy.signal import savgol_filter


@njit
def peak_peak_correlation_numba(scan_idx1, int_seq1, scan_idx2, int_seq2, roi_min_length=3):
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

    if n < roi_min_length:
        return 0.0

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

    return numerator / denominator if denominator != 0 else 1.0


@njit
def calc_ppc_numba(roi_ids, roi_rts, roi_scan_idx_seqs, roi_int_seqs, roi_lengths, rt_tol,
                   roi_min_length=3, min_ppc=0.8):
    n_rois = len(roi_ids)
    rows = []
    cols = []
    data = []

    for i in range(n_rois):
        start_i = np.sum(roi_lengths[:i])
        end_i = start_i + roi_lengths[i]
        for j in range(i + 1, n_rois):
            if abs(roi_rts[i] - roi_rts[j]) <= rt_tol:
                start_j = np.sum(roi_lengths[:j])
                end_j = start_j + roi_lengths[j]
                ppc = peak_peak_correlation_numba(
                    roi_scan_idx_seqs[start_i:end_i], roi_int_seqs[start_i:end_i],
                    roi_scan_idx_seqs[start_j:end_j], roi_int_seqs[start_j:end_j],
                    roi_min_length
                )
                if ppc >= min_ppc:
                    rows.append(i)
                    cols.append(j)
                    data.append(ppc)

    return np.array(rows), np.array(cols), np.array(data)


def calc_all_ppc(d, rt_tol=0.1, roi_min_length=3, min_ppc=0.8, save=True, path=None):
    """
    Calculate peak-peak correlation matrix for all ROIs in the dataset
    """
    rois = sorted(d.rois, key=lambda x: x.id)
    n_rois = len(rois)

    roi_ids = np.array([roi.id for roi in rois], dtype=np.int32)
    roi_rts = np.array([roi.rt for roi in rois], dtype=np.float64)
    roi_lengths = np.array([len(roi.scan_idx_seq) for roi in rois], dtype=np.int32)
    roi_scan_idx_seqs = np.concatenate([np.array(roi.scan_idx_seq, dtype=np.int32) for roi in rois])
    roi_int_seqs = np.concatenate([np.array(roi.int_seq, dtype=np.float64) for roi in rois])

    # smoothing
    # roi_int_seqs = np.concatenate([peak_smooth(np.array(roi.int_seq, dtype=np.float64)) for roi in rois])

    rows, cols, data = calc_ppc_numba(
        roi_ids, roi_rts, roi_scan_idx_seqs, roi_int_seqs, roi_lengths, rt_tol, roi_min_length, min_ppc
    )

    # Create sparse matrix
    ppc_matrix = csr_matrix((data, (rows, cols)), shape=(n_rois, n_rois), dtype=np.float64)
    ppc_matrix = ppc_matrix + ppc_matrix.T
    ppc_matrix.setdiag(1.0)

    if save and path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_npz(path, ppc_matrix)

    return ppc_matrix


def peak_smooth(int_array, window_length=5, poly_order=2):
    """
    Smoothing the intensity sequence using Savitzky-Golay filter.
    If the sequence is shorter than the window length, adjust the window size accordingly.

    Parameters:
    - int_array (array-like): The intensity sequence to smooth.
    - window_length (int): The length of the filter window (must be odd).
    - polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
    - smoothed_seq (numpy array): Smoothed version of the input intensity sequence.
    """
    int_len = len(int_array)

    if int_len <= poly_order:  # poly_order must be larger than 1
        return int_array

    # Ensure window_length is at most the length of int_seq and make it odd if necessary
    if window_length > int_len:
        window_length = int_len if int_len % 2 == 1 else int_len - 1

    return savgol_filter(int_array, window_length, poly_order)
