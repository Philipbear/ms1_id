"""
This module provides functions to preprocess MS2 spectra.
Some functions are adapted from ms_entropy package.
"""

import numpy as np


def preprocess_ms2(peaks, prec_mz,
                   min_mz=-1, max_mz=-1,
                   relative_intensity_cutoff=0.01,
                   min_ms2_difference_in_da=0.05,
                   min_ms2_difference_in_ppm=-1,
                   top6_every_50da=False,
                   peak_scale_k=None,
                   peak_intensity_power=1,
                   peak_norm=None) -> np.ndarray:
    """
    Main function to preprocess MS2 spectra.

        Clean, centroid, and normalize a spectrum with the following steps:
        1. Remove empty peaks (m/z <= 0 or intensity <= 0).
        2. Remove peaks with m/z >= max_mz or m/z < min_mz.
        3. Centroid the spectrum by merging peaks within min_ms2_difference_in_da.
        4. Remove peaks with intensity < relative_intensity_cutoff * max_intensity.
        5. Keep only the top max_peak_num peaks within every 50 Da.
        6. Transform the peak intensity.
        7. peak_intensity_power is used to transform the intensity to intensity^peak_intensity_power.
        8. Normalize the intensity.

        The cleaned spectrum will be sorted by m/z in ascending order.

    Parameters
    ----------
    peaks : np.ndarray in shape (n_peaks, 2)
        A 2D array of shape (n_peaks, 2) where the first column is m/z and the second column is intensity.

    prec_mz : float
        The precursor m/z value.

    min_mz : float, optional
        The minimum m/z to keep. Defaults to -1, which will skip removing peaks with m/z < min_mz.

    max_mz : float, optional
        The maximum m/z to keep. Defaults to -1, which will skip removing peaks with m/z >= max_mz.

    relative_intensity_cutoff : float, optional
        The minimum intensity to keep. Defaults to 0.01, which will remove peaks with intensity < 0.01 * max_intensity.

    min_ms2_difference_in_da : float, optional
        The minimum m/z difference between two peaks in the resulting spectrum. Defaults to 0.05, which will merge peaks within 0.05 Da. If a negative value is given, the min_ms2_difference_in_ppm will be used instead.

    min_ms2_difference_in_ppm : float, optional
        The minimum m/z difference between two peaks in the resulting spectrum. Defaults to -1, which will use the min_ms2_difference_in_da instead. If a negative value is given, the min_ms2_difference_in_da will be used instead.

    top6_every_50da : bool, optional
        Whether to keep only the top 6 peaks within every 50 Da. Defaults to False, which will keep all peaks.

    peak_intensity_power : float, optional
        The power to transform the peak intensity. Defaults to 1, which will not transform the intensity.

    peak_norm : str, optional
        The peak normalization method. The available methods are: 'sum', 'sum_sq'.
    """

    # Check the input spectrum and convert it to numpy array with shape (n, 2) and dtype np.float32.
    peaks = np.asarray(peaks, dtype=np.float32, order="C")
    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    assert peaks.ndim == 2 and peaks.shape[1] == 2, "The input spectrum must be a numpy array with shape (n, 2)."

    # Step 1. Remove empty peaks (m/z <= 0 or intensity <= 0).
    peaks = peaks[np.bitwise_and(peaks[:, 0] > 0, peaks[:, 1] > 0)]

    # Step 2. Remove peaks with m/z >= max_mz or m/z < min_mz.
    if min_mz is not None and min_mz > 0:
        peaks = peaks[peaks[:, 0] > min_mz]

    if max_mz is not None and max_mz > 0:
        peaks = peaks[peaks[:, 0] < max_mz]

    if peaks.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Step 3. Centroid the spectrum by merging peaks within min_ms2_difference_in_da.
    # The peaks will be sorted by m/z in ascending order.
    if min_ms2_difference_in_ppm > 0:
        peaks = centroid_spectrum(peaks, ms2_da=-1, ms2_ppm=min_ms2_difference_in_ppm)
    elif min_ms2_difference_in_da > 0:
        peaks = centroid_spectrum(peaks, ms2_da=min_ms2_difference_in_da, ms2_ppm=-1)
    else:
        pass

    # Step 4. Remove peaks with intensity < relative_intensity_cutoff * max_intensity.
    if relative_intensity_cutoff is not None:
        peaks = peaks[peaks[:, 1] >= relative_intensity_cutoff * np.max(peaks[:, 1])]

    # Step 5. Keep only the top 6 peaks within every 50 Da
    if top6_every_50da:
        peaks = top_n_per_mz_range(peaks, n_peaks=6, mz_range=50)

    # Step 6. scale the intensity.
    if peak_scale_k is not None:
        _prec_mz = prec_mz if prec_mz > 0 else np.max(peaks[:, 0])
        scaling_factor = peaks[:, 0] / _prec_mz * peak_scale_k
        peaks[:, 1] = peaks[:, 1] * np.exp(scaling_factor)

    # Step 7. Power the intensity.
    peaks[:, 1] = np.power(peaks[:, 1], peak_intensity_power)

    # Step 8. Normalize the intensity.
    if peak_norm is not None:
        if peak_norm == 'sum':
            peaks[:, 1] = peaks[:, 1] / np.sum(peaks[:, 1])
        elif peak_norm == 'sum_sq':
            peaks[:, 1] = peaks[:, 1] / np.sqrt(np.sum(peaks[:, 1] ** 2))

    return peaks


def top_n_per_mz_range(peaks: np.ndarray, n_peaks: int = 6, mz_range: int = 50) -> np.ndarray:
    """
    Keep only the top n_peaks peaks within every mz_range Da.
    """

    new_peaks = []  # List to store the filtered peaks
    # Calculate the range of m/z values
    min_mz_val, max_mz_val = np.min(peaks[:, 0]), np.max(peaks[:, 0])

    # Iterate over segments of 50 Da, starting from the minimum m/z value
    mz_start = 0  # Starting from 0 as mentioned
    while mz_start <= max_mz_val:
        mz_end = mz_start + mz_range
        # Filter peaks within the current m/z range
        segment_peaks = peaks[np.logical_and(peaks[:, 0] >= mz_start, peaks[:, 0] < mz_end)]

        # Check if there are any peaks in the segment
        if len(segment_peaks) > 0:
            # Sort the peaks by intensity in descending order to get the top peaks
            sorted_peaks = segment_peaks[np.argsort(-segment_peaks[:, 1])]
            # Keep only the top n_peaks, or all if there are fewer than n_peaks
            top_peaks = sorted_peaks[:min(n_peaks, len(sorted_peaks))]
            # Append the top peaks to the new_peaks list
            new_peaks.extend(top_peaks)

        mz_start += mz_range  # Move to the next segment

    # Convert the list back to a numpy array
    peaks = np.array(new_peaks)
    peaks = peaks[np.argsort(peaks[:, 0])]

    return peaks


def centroid_spectrum(peaks: np.ndarray, ms2_da: float = -1, ms2_ppm: float = -1) -> np.ndarray:
    """
    Centroid a spectrum by merging peaks within the +/- ms2_ppm or +/- ms2_da.

    Parameters
    ----------
    peaks : np.ndarray in shape (n_peaks, 2), dtype=np.float32
        A 2D array of shape (n_peaks, 2) where the first column is m/z and the second column is intensity.

    ms2_ppm : float, optional
        The m/z tolerance in ppm. Defaults to -1, which will use ms2_da instead.

    ms2_da : float, optional
        The m/z tolerance in Da. Defaults to -1, which will use ms2_ppm instead.

        **Note: ms2_ppm and ms2_da cannot be both negative, if so, an Exception will be raised. If both are positive, ms2_da will be used.**

    Returns
    -------
    np.ndarray in shape (n_peaks, 2), dtype=np.float32
        The resulting spectrum after centroiding have been sorted by m/z, and the difference between two peaks will be >= ms2_ppm or >= ms2_da.
    """
    peaks = np.asarray(peaks, dtype=np.float32, order="C")
    # Sort the peaks by m/z.
    peaks = peaks[np.argsort(peaks[:, 0])]
    is_centroided: int = _check_centroid(peaks, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
    while is_centroided == 0:
        peaks = _centroid_spectrum(peaks, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
        is_centroided = _check_centroid(peaks, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
    return peaks


def _centroid_spectrum(peaks: np.ndarray, ms2_da: float = -1, ms2_ppm: float = -1) -> np.ndarray:
    """Centroid a spectrum by merging peaks within the +/- ms2_ppm or +/- ms2_da."""
    # Construct a new spectrum to avoid memory reallocation.
    peaks_new = np.zeros((peaks.shape[0], 2), dtype=np.float32)
    peaks_new_len = 0

    # Get the intensity argsort order.
    intensity_order = np.argsort(peaks[:, 1])

    mz_delta_allowed_left = ms2_da
    mz_delta_allowed_right = ms2_da
    # Iterate through the peaks from high to low intensity.
    for idx in intensity_order[::-1]:
        if ms2_ppm > 0:
            mz_delta_allowed_left = peaks[idx, 0] * ms2_ppm * 1e-6
            # For the right boundary, the mz_delta_allowed_right = peaks[right_idx, 0] * ms2_ppm * 1e-6 = peaks[idx, 0] / (1 - ms2_ppm * 1e-6)
            mz_delta_allowed_right = peaks[idx, 0] / (1 - ms2_ppm * 1e-6)

        if peaks[idx, 1] > 0:
            # Find the left boundary.
            left_idx = idx - 1
            while left_idx >= 0 and peaks[idx, 0] - peaks[left_idx, 0] <= mz_delta_allowed_left:
                left_idx -= 1
            left_idx += 1

            # Find the right boundary.
            right_idx = idx + 1
            while right_idx < peaks.shape[0] and peaks[right_idx, 0] - peaks[idx, 0] <= mz_delta_allowed_right:
                right_idx += 1

            # Merge the peaks within the boundary. The new m/z is the weighted average of the m/z and intensity.
            intensity_sum = np.sum(peaks[left_idx:right_idx, 1])
            intensity_weighted_mz_sum = np.sum(peaks[left_idx:right_idx, 1] * peaks[left_idx:right_idx, 0])
            peaks_new[peaks_new_len, 0] = intensity_weighted_mz_sum / intensity_sum
            peaks_new[peaks_new_len, 1] = intensity_sum
            peaks_new_len += 1

            # Set the intensity of the merged peaks to 0.
            peaks[left_idx:right_idx, 1] = 0

    # Return the new spectrum.
    peaks_new = peaks_new[:peaks_new_len]
    return peaks_new[np.argsort(peaks_new[:, 0])]


def _check_centroid(peaks: np.ndarray, ms2_da: float = -1, ms2_ppm: float = -1) -> int:
    """Check if the spectrum is centroided. 0 for False and 1 for True."""
    if peaks.shape[0] <= 1:
        return 1

    if ms2_ppm >= 0:
        # Use ms2_ppm to check if the spectrum is centroided.
        return 1 if np.all(np.diff(peaks[:, 0]) >= peaks[1:, 0] * ms2_ppm * 1e-6) else 0
    elif ms2_da >= 0:
        # Use ms2_da to check if the spectrum is centroided.
        return 1 if np.all(np.diff(peaks[:, 0]) >= ms2_da) else 0
