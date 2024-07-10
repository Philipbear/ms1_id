import os

import numpy as np
from masscube.feature_grouping import peak_peak_correlation


def calc_all_ppc(d, rt_tol=0.1, save=True):
    """
    A function to calculate all peak-peak correlations

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.

    Returns
    ----------------------------------------------------------
    ppc_scores: n*3 numpy array
        (smaller id, larger id, ppc)
    """
    d.rois.sort(key=lambda x: x.id)
    rois = d.rois  # Assuming d.rois is a list of Roi objects

    # Pre-allocate a list to store the results
    ppc_list = []

    # loop
    for i, roi1 in enumerate(rois):
        for roi2 in rois[i + 1:]:
            if abs(roi1.rt - roi2.rt) <= rt_tol:
                ppc = peak_peak_correlation(roi1, roi2)
                if ppc != 0:  # Only store non-zero correlations
                    ppc_list.append([roi1.id, roi2.id, ppc])

    # Convert the list to a numpy array
    ppc_array = np.array(ppc_list)

    if len(ppc_array) > 0:
        # Sort the array by the first column (smaller id), then by the second column (larger id)
        ppc_array = ppc_array[np.lexsort((ppc_array[:, 1], ppc_array[:, 0]))]

    if save:
        # save path
        path = os.path.join(d.params.single_file_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(d.params.single_file_dir, d.file_name + "_peakCor.npy")

        # save as npy
        np.save(path, ppc_array)

    return
