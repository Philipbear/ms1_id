import numpy as np
from numba import njit


def centroid_spectrum(mz_list, intensity_list, centroid_mode='max',
                      ms2_da=0.005, ms2_ppm=25.0):
    """
    Centroid a spectrum by merging peaks within the +/- ms2_ppm or +/- ms2_da.
    centroid_mode: 'max' or 'sum'
    """
    # Ensure mz_list and intensity_list are numpy arrays
    mz_list = np.asarray(mz_list, dtype=np.float32)
    intensity_list = np.asarray(intensity_list, dtype=np.float32)

    # Create a 2D array of peaks
    peaks = np.column_stack((mz_list, intensity_list))

    # Sort the peaks by m/z
    peaks = peaks[np.argsort(peaks[:, 0])]

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
    # ... (rest of the function remains the same)


@njit
def check_centroid(peaks, ms2_da=0.005, ms2_ppm=25.0):
    """Check if the spectrum is centroided. Returns True if centroided, False otherwise."""
    if peaks.shape[0] <= 1:
        return True

    # Calculate ms2_ppm_in_da
    ms2_ppm_in_da = peaks[1:, 0] * ms2_ppm * 1e-6

    # Use bitwise OR to choose the maximum
    threshold = np.maximum(ms2_da, ms2_ppm_in_da)

    # Check if the spectrum is centroided
    return np.all(np.diff(peaks[:, 0]) >= threshold)