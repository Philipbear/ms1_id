import numpy as np
from numba import njit


def centroid_spectrum(mz_list, intensity_list, centroid_mode='max',
                      ms2_da=0.005, ms2_ppm=25.0):
    """
    Centroid a spectrum by merging peaks within the +/- ms2_ppm or +/- ms2_da.
    centroid_mode: 'max' or 'sum'
    """
    # Ensure mz_list and intensity_list are contiguous numpy arrays
    mz_list = np.ascontiguousarray(mz_list, dtype=np.float32)
    intensity_list = np.ascontiguousarray(intensity_list, dtype=np.float32)

    # Create a 2D array of peaks
    peaks = np.column_stack((mz_list, intensity_list))

    # Sort the peaks by m/z
    peaks = np.ascontiguousarray(peaks[np.argsort(peaks[:, 0])])

    is_centroided = check_centroid(peaks, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
    while not is_centroided:
        peaks = centroid_spectrum_inner(peaks, centroid_mode, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
        is_centroided = check_centroid(peaks, ms2_da=ms2_da, ms2_ppm=ms2_ppm)

    return peaks[:, 0], peaks[:, 1]


@njit
def centroid_spectrum_inner(peaks, centroid_mode='max',
                            ms2_da=0.005, ms2_ppm=25.0):
    """Centroid a spectrum by merging peaks within the +/- ms2_ppm or +/- ms2_da,
    only merging peaks with lower intensity than the target peak."""
    # Ensure peaks is contiguous
    peaks = np.ascontiguousarray(peaks)

    # Construct a new spectrum to avoid memory reallocation.
    peaks_new = np.zeros((peaks.shape[0], 2), dtype=np.float32)
    peaks_new_len = 0

    # Get the intensity argsort order.
    intensity_order = np.argsort(peaks[:, 1])

    # Iterate through the peaks from high to low intensity.
    for idx in intensity_order[::-1]:
        if peaks[idx, 1] > 0:
            mz_delta_allowed_ppm_in_da = peaks[idx, 0] * ms2_ppm * 1e-6
            mz_delta_allowed = max(ms2_da, mz_delta_allowed_ppm_in_da)

            # Find indices of peaks to merge.
            indices = np.where(np.logical_and(np.abs(peaks[:, 0] - peaks[idx, 0]) <= mz_delta_allowed,
                                              peaks[:, 1] < peaks[idx, 1]))[0]

            if centroid_mode == 'max':
                # Merge the peaks
                peaks_new[peaks_new_len, 0] = peaks[idx, 0]
                peaks_new[peaks_new_len, 1] = peaks[idx, 1]
                peaks_new_len += 1

                peaks[indices, 1] = 0

            elif centroid_mode == 'sum':
                # Merge the peaks
                intensity_sum = peaks[idx, 1]
                intensity_weighted_mz_sum = peaks[idx, 1] * peaks[idx, 0]
                for i in indices:
                    intensity_sum += peaks[i, 1]
                    intensity_weighted_mz_sum += peaks[i, 1] * peaks[i, 0]
                    peaks[i, 1] = 0  # Set the intensity of the merged peaks to 0

                peaks_new[peaks_new_len, 0] = intensity_weighted_mz_sum / intensity_sum
                peaks_new[peaks_new_len, 1] = intensity_sum
                peaks_new_len += 1

                # Set the intensity of the target peak to 0
                peaks[idx, 1] = 0

    # Return the new spectrum.
    peaks_new = peaks_new[:peaks_new_len]
    return np.ascontiguousarray(peaks_new[np.argsort(peaks_new[:, 0])])


@njit
def check_centroid(peaks, ms2_da=0.005, ms2_ppm=25.0):
    """Check if the spectrum is centroided. Returns True if centroided, False otherwise."""
    # Ensure peaks is contiguous
    peaks = np.ascontiguousarray(peaks)

    if peaks.shape[0] <= 1:
        return True

    # Calculate ms2_ppm_in_da
    ms2_ppm_in_da = peaks[1:, 0] * ms2_ppm * 1e-6

    # Use maximum to choose the larger value
    threshold = np.maximum(ms2_da, ms2_ppm_in_da)

    # Check if the spectrum is centroided
    return np.all(np.diff(peaks[:, 0]) >= threshold)