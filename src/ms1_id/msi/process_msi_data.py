import multiprocessing
import os
import pickle
from collections import defaultdict

import numpy as np
import pyimzml.ImzMLParser as imzml
from numba import njit
from tqdm import tqdm

from ms1_id.msi.centroid_data import centroid_spectrum


def process_ms_imaging_data(imzml_file, ibd_file, mass_detect_int_tol=None,
                            mz_bin_size=0.005,
                            noise_detection='moving_average', sn_factor=5.0,
                            centroided=False,
                            n_processes=None,
                            save=False, save_dir=None):
    validate_inputs(noise_detection)

    parser = imzml.ImzMLParser(imzml_file)

    # Check if results already exist
    if save_dir and check_existing_results(save_dir):
        return load_existing_results(save_dir)

    # Auto-detect intensity threshold if not provided
    if mass_detect_int_tol is None:
        print(f'Auto-denoising MS spectra using {noise_detection} method.')
        if noise_detection == 'percentile':
            mass_detect_int_tol = detect_threshold_percentile(parser, sn_factor)

    mz_intensity_dict, coordinates = process_spectra(parser, noise_detection, mass_detect_int_tol,
                                                     mz_bin_size, sn_factor, centroided, n_processes)

    mz_values, intensity_matrix = convert_to_arrays(mz_intensity_dict, coordinates)

    # Save results if requested
    if save and save_dir:
        print(f'Saving mz values, intensity matrix, and coordinates...')
        save_results(save_dir, mz_values, intensity_matrix, coordinates)

    return mz_values, intensity_matrix, coordinates, parser.polarity


def validate_inputs(noise_detection):
    assert noise_detection in ['percentile', 'moving_average'], \
        "Noise reduction method must be 'percentile' or 'moving_average'"


def check_existing_results(save_dir):
    mz_values_path = os.path.join(save_dir, 'mz_values.npy')
    intensity_matrix_path = os.path.join(save_dir, 'intensity_matrix.npy')
    coordinates_path = os.path.join(save_dir, 'coordinates.pkl')
    return all(os.path.exists(path) for path in [mz_values_path, intensity_matrix_path, coordinates_path])


def load_existing_results(save_dir):
    mz_values = np.load(os.path.join(save_dir, 'mz_values.npy'))
    intensity_matrix = np.load(os.path.join(save_dir, 'intensity_matrix.npy'))
    with open(os.path.join(save_dir, 'coordinates.pkl'), 'rb') as f:
        coordinates = pickle.load(f)
    return mz_values, intensity_matrix, coordinates, None  # Note: parser.polarity is not saved


def detect_threshold_percentile(parser, sn_factor=5.0):
    all_intensities = []
    for idx, _ in enumerate(parser.coordinates):
        _, intensity = parser.getspectrum(idx)
        all_intensities.extend(intensity)

    all_intensities = np.array(all_intensities)
    non_zero_intensities = all_intensities[all_intensities > 0.0]
    threshold = max(np.min(non_zero_intensities) * sn_factor, np.percentile(non_zero_intensities, 10))
    print('Auto-detected intensity threshold (percentile method):', threshold)
    return threshold


def process_spectra(parser, noise_detection, mass_detect_int_tol, mz_bin_size, sn_factor, centroided, n_processes):
    args_list = [(idx, *parser.getspectrum(idx), noise_detection, mass_detect_int_tol,
                  mz_bin_size, sn_factor, centroided)
                 for idx, _ in enumerate(parser.coordinates)]

    n_processes = n_processes or multiprocessing.cpu_count()

    if n_processes == 1:
        # Non-parallel processing
        results = []
        for args in tqdm(args_list, desc="Denoising spectra", unit="spectrum"):
            results.append(process_spectrum(args))
    else:
        # Parallel processing
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = list(tqdm(pool.imap(process_spectrum, args_list),
                                total=len(args_list),
                                desc="Denoising spectra",
                                unit="spectrum"))

    mz_intensity_dict = defaultdict(lambda: defaultdict(float))
    coordinates = []
    for idx, binned_data in results:
        for binned_m, summed_intensity in binned_data.items():
            mz_intensity_dict[binned_m][idx] = summed_intensity
        coordinates.append(parser.coordinates[idx][:2])  # Only x and y

    return mz_intensity_dict, coordinates


def convert_to_arrays(mz_intensity_dict, coordinates):
    mz_values = np.array(sorted(mz_intensity_dict.keys()))
    intensity_matrix = np.array([
        [mz_intensity_dict[mz].get(idx, 0.0) for idx in range(len(coordinates))]
        for mz in mz_values
    ])
    return mz_values, intensity_matrix


def save_results(save_dir, mz_values, intensity_matrix, coordinates):
    np.save(os.path.join(save_dir, 'mz_values.npy'), mz_values)
    np.save(os.path.join(save_dir, 'intensity_matrix.npy'), intensity_matrix)
    with open(os.path.join(save_dir, 'coordinates.pkl'), 'wb') as f:
        pickle.dump(coordinates, f)


def process_spectrum(args):
    idx, mz, intensity, noise_detection, mass_detect_int_tol, mz_bin_size, sn_factor, centroided = args

    if noise_detection == 'moving_average':
        baseline = moving_average_baseline(mz, intensity, factor=sn_factor)
        mask = intensity > baseline
    else:
        mask = intensity > mass_detect_int_tol

    # Filter mz and intensity arrays
    filtered_mz = mz[mask]
    filtered_intensity = intensity[mask]

    # Centroid the spectrum
    if not centroided:
        filtered_mz, filtered_intensity = centroid_spectrum(filtered_mz, filtered_intensity,
                                                            centroid_mode='max',
                                                            width_da=0.005, width_ppm=25)

    # Bin m/z values and take max intensity within each bin
    binned_data = {}
    for m, i in zip(filtered_mz, filtered_intensity):
        binned_m = round(m / mz_bin_size) * mz_bin_size
        if binned_m not in binned_data or i > binned_data[binned_m]:
            binned_data[binned_m] = i

    return idx, binned_data


@njit
def moving_average_baseline(mz_array, intensity_array,
                            mz_window=100.0, percentage_lowest=0.05, factor=5.0):
    """
    Apply moving average algorithm to a single mass spectrum using an m/z-based window.
    This function is optimized with Numba.

    :param mz_array: numpy array of m/z values
    :param intensity_array: numpy array of intensity values
    :param mz_window: size of the moving window in Da (default: 100.0)
    :param percentage_lowest: percentage of lowest points to consider in each window (default: 5%)
    :param factor: factor to multiply the mean of the lowest points (default: 5.0)
    :return: tuple of (filtered_mz_array, filtered_intensity_array)
    """
    baseline_array = []

    for i in range(len(mz_array)):
        mz_min = mz_array[i] - mz_window / 2
        mz_max = mz_array[i] + mz_window / 2

        window_intensities = intensity_array[(mz_array >= mz_min) & (mz_array <= mz_max)]

        if len(window_intensities) > 0:
            positive_intensities = window_intensities[window_intensities > 0]
            if len(positive_intensities) > 0:
                sorted_intensities = np.sort(positive_intensities)
                n_lowest = max(1, int(len(sorted_intensities) * percentage_lowest))
                lowest_n = sorted_intensities[:n_lowest]
                baseline = factor * np.mean(lowest_n)
            else:
                baseline = 0.0
        else:
            baseline = 0.0

        baseline_array.append(baseline)

    return np.array(baseline_array)
