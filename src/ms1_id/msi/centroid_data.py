"""
This file contains code modified from the FlashEntropySearch project
(https://github.com/YuanyueLi/FlashEntropySearch)
Copyright Yuanyue Li in 2023

Modified by Shipei Xing in 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from numba import njit


def centroid_spectrum(mz_list, intensity_list, centroid_mode='max',
                      width_da=0.005, width_ppm=25.0):
    """
    Centroid a spectrum by merging peaks within the +/- ms2_ppm or +/- ms2_da.
    centroid_mode: 'max' or 'sum'
    """
    peaks = list(zip(mz_list, intensity_list))
    peaks = np.asarray(peaks, dtype=np.float32, order="C")

    if len(peaks) == 0:
        return mz_list, intensity_list

    # Sort the peaks by m/z.
    peaks = peaks[np.argsort(peaks[:, 0])]

    # centroid the spectrum
    peaks = _centroid_spectrum(peaks, centroid_mode, width_da=width_da, width_ppm=width_ppm)
    return peaks[:, 0], peaks[:, 1]


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


def _check_centroid(peaks, width_da=0.005, width_ppm=25.0) -> int:
    """Check if the spectrum is centroided. 0 for False and 1 for True."""
    if peaks.shape[0] <= 1:
        return 1

    # Calculate width_ppm_in_da
    width_ppm_in_da = peaks[1:, 0] * width_ppm * 1e-6

    # Use bitwise OR to choose the maximum
    threshold = np.maximum(width_da, width_ppm_in_da)

    # Check if the spectrum is centroided
    return 1 if np.all(np.diff(peaks[:, 0]) >= threshold) else 0


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
