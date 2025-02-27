import multiprocessing
import os
import pickle

import numpy as np
import pyimzml.ImzMLParser as imzml
from tqdm import tqdm

from ms1_id.msi.utils_imaging import MsiFeature
from ms1_id.msi.msi_raw_data_utils import clean_msi_spec, get_msi_features, assign_spec_to_feature_array, calc_spatial_chaos


def process_ms_imaging_data(imzml_file, ibd_file,
                            polarity=None,
                            mz_ppm_tol=10.0,
                            sn_factor=5.0,
                            min_feature_pixel_count=50,
                            min_feature_spatial_chaos=0.01,
                            n_processes=None,
                            save_dir=None):

    parser = imzml.ImzMLParser(imzml_file)

    # Set polarity
    file_polarity = determine_file_polarity(parser, polarity)

    # Check if results already exist
    if save_dir and check_existing_results(save_dir):
        feature_ls, intensity_matrix = load_existing_results(save_dir)
        return feature_ls, intensity_matrix, file_polarity

    centroided = determine_centroid(imzml_file)

    if not centroided:
        print("Input data are not centroided, centroiding spectra...")

    # Get features (list of MsiFeature)
    feature_ls, intensity_matrix = process_spectra(parser, mz_ppm_tol, sn_factor, centroided, min_feature_pixel_count, n_processes)

    # Filter features based on spatial chaos
    feature_ls, intensity_matrix = filter_features_by_spatial_chaos(feature_ls, intensity_matrix, parser.coordinates,
                                                                    min_feature_spatial_chaos, n_processes)

    # Save
    if save_dir:
        print(f'Saving features and intensity matrix...')
        save_results(save_dir, feature_ls, intensity_matrix)

    return feature_ls, intensity_matrix, file_polarity


def determine_file_polarity(parser, file_polarity):
    if parser.polarity in ['positive', 'negative']:
        return parser.polarity

    if file_polarity is not None:
        if file_polarity.lower() in ['positive', 'pos', 'p']:
            file_polarity = 'positive'
        elif file_polarity.lower() in ['negative', 'neg', 'n']:
            file_polarity = 'negative'
        else:
            file_polarity = None

    return file_polarity


def check_existing_results(save_dir):
    mz_values_path = os.path.join(save_dir, 'mz_values.npy')
    intensity_matrix_path = os.path.join(save_dir, 'intensity_matrix.npy')
    return all(os.path.exists(path) for path in [mz_values_path, intensity_matrix_path])


def load_existing_results(save_dir):
    with open(os.path.join(save_dir, 'features.pkl'), 'rb') as f:
        feature_ls = pickle.load(f)
    intensity_matrix = np.load(os.path.join(save_dir, 'intensity_matrix.npy'))
    return feature_ls, intensity_matrix


def process_spectra(parser, mz_ppm_tol, sn_factor, centroided, min_feature_pixel_count, n_processes):
    # first get all spectra and clean them
    args_list = [(idx, *parser.getspectrum(idx), sn_factor, centroided)
                 for idx, _ in enumerate(parser.coordinates)]

    if n_processes == 1:
        # Non-parallel processing
        results = []
        for args in tqdm(args_list, desc="Denoising spectra", unit="spectrum"):
            results.append(clean_msi_spectrum(args))
    else:
        # Parallel processing with chunks
        chunk_size = min(200, len(parser.coordinates) // n_processes + 1)
        arg_chunks = chunk_list(args_list, chunk_size)

        with multiprocessing.Pool(processes=n_processes) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_spectrum_chunk, arg_chunks),
                total=len(arg_chunks),
                desc="Denoising spectrum chunks",
                unit="chunk"
            ))
            results = [item for sublist in chunk_results for item in sublist]

    # get features from all mzs and intensities
    print("Getting features from all m/z values and intensities...")
    all_mzs = []
    all_intensities = []
    for idx, mz_arr, intensity_arr in results:
        all_mzs.extend(mz_arr.tolist())
        all_intensities.extend(intensity_arr.tolist())
    feature_mzs = get_msi_features(np.array(all_mzs), np.array(all_intensities), mz_ppm_tol,
                                   min_group_size=min_feature_pixel_count,
                                   n_processes=n_processes)
    print(f"Found {len(feature_mzs)} features.")
    del all_mzs, all_intensities

    # assign mzs to features
    args_list = [(idx, *parser.getspectrum(idx), sn_factor, centroided, feature_mzs, mz_ppm_tol)
                 for idx, _ in enumerate(parser.coordinates)]

    # Initialize intensity matrix
    n_coordinates = len(parser.coordinates)
    intensity_matrix = np.zeros((len(feature_mzs), n_coordinates))

    if n_processes == 1:
        # Non-parallel processing - keep original behavior
        for args in tqdm(args_list, desc="Feature detection", unit="spectrum"):
            idx, intensities = assign_spectrum_to_feature(args)
            intensity_matrix[:, idx] = intensities
    else:
        # Parallel processing with chunks
        chunk_size = min(200, len(parser.coordinates) // n_processes + 1)
        arg_chunks = chunk_list(args_list, chunk_size)

        with multiprocessing.Pool(processes=n_processes) as pool:
            for chunk_results in tqdm(
                    pool.imap(assign_spectrum_chunk, arg_chunks),
                    total=len(arg_chunks),
                    desc="Processing feature chunks",
                    unit="chunk"
            ):
                for idx, intensities in chunk_results:
                    intensity_matrix[:, idx] = intensities

    # Create MsiFeature objects
    feature_ls = [MsiFeature(i, mz, None, None) for i, mz in enumerate(feature_mzs)]

    # # Assign feature intensities
    # for idx in range(len(feature_ls)):
    #     feature_ls[idx].intensity_arr = intensity_matrix[idx, :]

    return feature_ls, intensity_matrix


def filter_features_by_spatial_chaos(feature_ls, intensity_matrix, coordinates,
                                     min_feature_spatial_chaos, n_processes):
    """
    Filter features based on their spatial chaos score.

    Parameters:
    -----------
    feature_ls : list of MsiFeature
    intensity_matrix : numpy.ndarray
        Matrix of intensities for each feature at each coordinate
    coordinates : list
        List of (x, y, z) coordinates
    min_feature_spatial_chaos : float
        Minimum spatial chaos score to keep a feature
    n_processes : int, optional
        Number of processes to use for parallel processing

    Returns:
    --------
    numpy.ndarray
        Filtered array of m/z values
    numpy.ndarray
        Filtered intensity matrix
    """

    # Calculate spatial chaos for each feature
    print(f"Calculating spatial chaos for {len(feature_ls)} features...")

    if n_processes == 1:
        all_results = []
        # Non-parallel processing
        for idx in tqdm(range(len(feature_ls)), desc="Calculating spatial chaos", unit="feature"):
            chaos = calc_spatial_chaos(intensity_matrix[idx], coordinates)
            all_results.append((idx, chaos))
    else:
        # Parallel processing
        args_list = [(idx, intensity_matrix[idx], coordinates)
                     for idx in range(len(feature_ls))]

        # Split into chunks for better progress tracking
        chunk_size = min(500, len(feature_ls) // n_processes + 1)
        arg_chunks = chunk_list(args_list, chunk_size)

        with multiprocessing.Pool(processes=n_processes) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_spatial_chaos_chunk, arg_chunks),
                total=len(arg_chunks),
                desc="Calculating spatial chaos chunks",
                unit="chunk"
            ))
            all_results = [item for sublist in chunk_results for item in sublist]

    # Filter features based on spatial chaos
    indices_to_keep = []
    new_feature_ls = []
    new_idx = 0
    for idx, chaos in all_results:
        if chaos > min_feature_spatial_chaos:
            indices_to_keep.append(idx)
            new_feature_ls.append(MsiFeature(new_idx, feature_ls[idx].mz, None, chaos))
            new_idx += 1

    # Filter intensity matrix
    filtered_intensity_matrix = intensity_matrix[indices_to_keep, :]

    print(f"Kept {len(new_feature_ls)} out of {len(feature_ls)} features after spatial chaos filtering.")

    return new_feature_ls, filtered_intensity_matrix


def process_spatial_chaos_chunk(chunk_args):
    """Process a chunk of features for spatial chaos calculation."""
    chunk_results = []
    for args in chunk_args:
        feature_idx, feature_intensity_array, coordinates = args
        chaos = calc_spatial_chaos(feature_intensity_array, coordinates)
        chunk_results.append((feature_idx, chaos))
    return chunk_results


def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_spectrum_chunk(chunk_args):
    """Process a chunk of spectra."""
    chunk_results = []
    for args in chunk_args:
        idx, mz, intensity, sn_factor, centroided = args
        filtered_mz, filtered_intensity = clean_msi_spec(mz, intensity, sn_factor, centroided)
        chunk_results.append((idx, filtered_mz, filtered_intensity))
    return chunk_results


def assign_spectrum_chunk(chunk_args):
    """Assign features for a chunk of spectra."""
    chunk_results = []
    for args in chunk_args:
        idx, mz, intensity, sn_factor, centroided, feature_mzs, mz_ppm_tol = args
        filtered_mz, filtered_intensity = clean_msi_spec(mz, intensity, sn_factor, centroided)
        feature_intensities = assign_spec_to_feature_array(filtered_mz, filtered_intensity, feature_mzs, mz_ppm_tol)
        chunk_results.append((idx, feature_intensities))
    return chunk_results


def clean_msi_spectrum(args):
    idx, mz, intensity, sn_factor, centroided = args

    mz, intensity = clean_msi_spec(mz, intensity, sn_factor, centroided)

    return idx, mz, intensity


def assign_spectrum_to_feature(args):
    idx, mz, intensity, sn_factor, centroided, feature_mzs, mz_ppm_tol = args

    mz, intensity = clean_msi_spec(mz, intensity, sn_factor, centroided)

    feature_intensities = assign_spec_to_feature_array(mz, intensity, feature_mzs, mz_ppm_tol)

    return idx, feature_intensities


def save_results(save_dir, feature_ls, intensity_matrix):
    """Save the results to the specified directory."""
    with open(os.path.join(save_dir, 'features.pkl'), 'wb') as f:
        pickle.dump(feature_ls, f)
    np.save(os.path.join(save_dir, 'intensity_matrix.npy'), intensity_matrix)


def determine_centroid(imzml_file):
    centroid = False
    with open(imzml_file, 'r') as f:
        for i, line in enumerate(f):
            if "centroid spectrum" in line.lower():
                centroid = True
            if i > 300:
                break

    return centroid


if __name__ == '__main__':
    # Example usage
    imzml_file = '/Users/shipei/Documents/projects/ms1_id/imaging/mouse_body/wb xenograft in situ metabolomics test - rms_corrected.imzML'

    parser = imzml.ImzMLParser(imzml_file)
    print(parser.polarity)
