from masscube.feature_grouping import peak_peak_correlation
import os
from scipy.sparse import csr_matrix, save_npz, load_npz


def calc_all_ppc(d, rt_tol=0.1, save=True, ppc_threshold=0.0):
    """
    A function to calculate all peak-peak correlations and store them efficiently
    :param d: MSData object
    :param rt_tol: retention time tolerance for aligning features
    :param save: whether to save the results
    :param ppc_threshold: minimum PPC score to keep
    """
    rois = sorted(d.rois, key=lambda x: x.id)
    n_rois = len(rois)

    # Create a sparse matrix to store PPC scores
    ppc_matrix = csr_matrix((n_rois, n_rois))

    for i, roi1 in enumerate(rois):
        for j, roi2 in enumerate(rois[i + 1:], start=i + 1):
            if abs(roi1.rt - roi2.rt) <= rt_tol:
                ppc = peak_peak_correlation(roi1, roi2)
                if ppc > ppc_threshold:
                    ppc_matrix[roi1.id, roi2.id] = ppc
                    ppc_matrix[roi2.id, roi1.id] = ppc  # Symmetric matrix

    if save:
        path = os.path.join(d.params.single_file_dir, d.file_name + "_peakCor.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_npz(path, ppc_matrix)

    return ppc_matrix
