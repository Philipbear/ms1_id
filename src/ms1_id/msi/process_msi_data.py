import multiprocessing
import os
import pickle
from collections import defaultdict

import numpy as np
import pyimzml.ImzMLParser as imzml
from numba import njit
from tqdm import tqdm

from ms1_id.msi.centroid_data import centroid_spectrum
from ms1_id.msi.msi_feature_extraction import get_msi_features


def process_ms_imaging_data(imzml_file, ibd_file,
                            mz_ppm_tol=5,
                            sn_factor=5.0,
                            n_processes=None,
                            save=False, save_dir=None):

    parser = imzml.ImzMLParser(imzml_file)

    # Check if results already exist
    if save_dir and check_existing_results(save_dir):
        return load_existing_results(save_dir)

    centroided = determine_centroid(imzml_file)

    feature_mzs, intensity_matrix, coordinates = process_spectra(parser, mz_ppm_tol, sn_factor, centroided, n_processes)

    # Save results if requested
    if save and save_dir:
        print(f'Saving mz values, intensity matrix, and coordinates...')
        save_results(save_dir, feature_mzs, intensity_matrix, coordinates)

    return feature_mzs, intensity_matrix, coordinates, parser.polarity


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


def process_spectra(parser, mz_ppm_tol, sn_factor, centroided, n_processes):

    # first get all spectra and clean them
    args_list = [(idx, *parser.getspectrum(idx), mz_ppm_tol, sn_factor, centroided)
                 for idx, _ in enumerate(parser.coordinates)]

    n_processes = n_processes or multiprocessing.cpu_count()

    if n_processes == 1:
        # Non-parallel processing
        results = []
        for args in tqdm(args_list, desc="Denoising spectra", unit="spectrum"):
            results.append(clean_msi_spectrum(args))
    else:
        # Parallel processing
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = list(tqdm(pool.imap(clean_msi_spectrum, args_list),
                                total=len(args_list), desc="Denoising spectra", unit="spectrum"))

    # get all mzs and intensities
    all_mzs = []
    all_intensities = []
    idx_mz_intensity_dict = {}  # no need to re-clean the spectra
    for idx, mz_arr, intensity_arr in results:
        idx_mz_intensity_dict[idx] = mz_arr, intensity_arr
        all_mzs.extend(mz_arr.tolist())
        all_intensities.extend(intensity_arr.tolist())

    # get features from all mzs and intensities
    feature_mzs = get_msi_features(np.array(all_mzs), np.array(all_intensities), mz_ppm_tol)
    del all_mzs, all_intensities

    # assign mzs to features
    args_list = [(idx, idx_mz_intensity_dict[idx], feature_mzs)
                 for idx, _ in enumerate(parser.coordinates)]
    # assign mzs to features
    if n_processes == 1:
        results = []
        for args in tqdm(args_list, desc="Feature detection", unit="spectrum"):
            results.append(clean_msi_spectrum(args))
    else:
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = list(tqdm(pool.imap(clean_msi_spectrum, args_list),
                                total=len(args_list), desc="Feature detection", unit="spectrum"))

    # Create intensity matrix
    feature_mzs = np.array(sorted(feature_mzs))
    n_coordinates = len(parser.coordinates)
    intensity_matrix = np.zeros((len(feature_mzs), n_coordinates))
    coordinates = []

    for idx, feature_intensity_dict in results:
        coordinates.append(parser.coordinates[idx][:2])  # Only x and y
        for feature_idx, mz in enumerate(feature_mzs):
            if mz in feature_intensity_dict:
                intensity_matrix[feature_idx, idx] = feature_intensity_dict[mz]

    return feature_mzs, intensity_matrix, coordinates


def save_results(save_dir, mz_values, intensity_matrix, coordinates):
    np.save(os.path.join(save_dir, 'mz_values.npy'), mz_values)
    np.save(os.path.join(save_dir, 'intensity_matrix.npy'), intensity_matrix)
    with open(os.path.join(save_dir, 'coordinates.pkl'), 'wb') as f:
        pickle.dump(coordinates, f)


def clean_msi_spectrum(args):
    idx, mz, intensity, mz_bin_size, sn_factor, centroided = args

    baseline = moving_average_baseline(mz, intensity, factor=sn_factor)
    mask = intensity > baseline

    # Filter mz and intensity arrays
    filtered_mz = mz[mask]
    filtered_intensity = intensity[mask]

    # Centroid the spectrum
    if not centroided:
        filtered_mz, filtered_intensity = centroid_spectrum(filtered_mz, filtered_intensity,
                                                            centroid_mode='max',
                                                            width_da=0.002, width_ppm=10)

    return idx, filtered_mz, filtered_intensity


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


def determine_centroid(imzml_file):

    centroid = False

    with open(imzml_file, 'r') as f:
        for i, line in enumerate(f):
            if "centroid spectrum" in line.lower():
                centroid = True
            if i > 300:
                break

    return centroid