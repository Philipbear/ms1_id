import numpy as np
from numba import njit


@njit
def _centroid_spectrum(peaks, centroid_mode='max', peak_height_requirement=True,
                       width_da=0.005, width_ppm=25.0) -> np.ndarray:
    """Centroid a spectrum by merging peaks within the +/- ms2_ppm or +/- ms2_da,
    only merging peaks with lower intensity than the target peak."""
    # Construct a new spectrum to avoid memory reallocation.
    peaks_new = np.zeros((peaks.shape[0], 2), dtype=np.float32)
    peaks_new_len = 0

    # Get the intensity argsort order.
    intensity_order = np.argsort(peaks[:, 1])

    # Iterate through the peaks from high to low intensity.
    for idx in intensity_order[::-1]:
        if peaks[idx, 1] > 0:
            mz_delta_allowed_ppm_in_da = peaks[idx, 0] * width_ppm * 1e-6
            mz_delta_allowed = max(width_da, mz_delta_allowed_ppm_in_da)

            # Find indices of peaks to merge.
            if peak_height_requirement:
                indices = np.where(np.logical_and(np.abs(peaks[:, 0] - peaks[idx, 0]) <= mz_delta_allowed,
                                                  peaks[:, 1] < peaks[idx, 1]))[0]
            else:
                indices = np.where(np.abs(peaks[:, 0] - peaks[idx, 0]) <= mz_delta_allowed)[0]

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
    return peaks_new[np.argsort(peaks_new[:, 0])]


def check_centroid_for_search(peaks, width_da=0.105):
    """Check if the spectrum is centroided. True for centroided and False for not centroided."""

    # Check if the spectrum is centroided
    return True if np.all(np.diff(peaks[:, 0]) >= width_da) else False


def centroid_spectrum_for_search(peaks, width_da=0.105):
    """Centroid a spectrum for search."""
    if len(peaks) == 0:
        return peaks

    if check_centroid_for_search(peaks, width_da=width_da):
        return peaks

    # Sort the peaks by m/z.
    peaks = peaks[np.argsort(peaks[:, 0])]

    # centroid the spectrum
    peaks = _centroid_spectrum(peaks, centroid_mode='max', width_da=width_da)
    return peaks