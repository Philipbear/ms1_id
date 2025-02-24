import multiprocessing
import os
import pickle

import numpy as np
import pyimzml.ImzMLParser as imzml
from tqdm import tqdm


from ms1_id.msi.msi_raw_data_utils import clean_msi_spec, get_msi_features, assign_spec_to_feature_array


def process_ms_imaging_data(imzml_file, ibd_file,
                            mz_ppm_tol=10.0,
                            sn_factor=5.0,
                            n_processes=None,
                            save_dir=None):
    parser = imzml.ImzMLParser(imzml_file)

    # Check if results already exist
    if save_dir and check_existing_results(save_dir):
        mz_values, intensity_matrix, coordinates = load_existing_results(save_dir)
        return mz_values, intensity_matrix, coordinates, parser.polarity

    centroided = determine_centroid(imzml_file)

    if not centroided:
        print("Input data are not centroided, centroiding spectra...")

    feature_mzs, intensity_matrix, coordinates = process_spectra(parser, mz_ppm_tol, sn_factor, centroided, n_processes)

    # Save
    if save_dir:
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
    return mz_values, intensity_matrix, coordinates


# def process_spectra(parser, mz_ppm_tol, sn_factor, centroided, n_processes):
#     # first get all spectra and clean them
#     args_list = [(idx, *parser.getspectrum(idx), sn_factor, centroided)
#                  for idx, _ in enumerate(parser.coordinates)]
#
#     n_processes = n_processes or multiprocessing.cpu_count()
#
#     if n_processes == 1:
#         # Non-parallel processing
#         results = []
#         for args in tqdm(args_list, desc="Denoising spectra", unit="spectrum"):
#             results.append(clean_msi_spectrum(args))
#     else:
#         # Parallel processing
#         with multiprocessing.Pool(processes=n_processes) as pool:
#             results = list(tqdm(pool.imap(clean_msi_spectrum, args_list),
#                                 total=len(args_list), desc="Denoising spectra", unit="spectrum"))
#
#     # get all mzs and intensities
#     all_mzs = []
#     all_intensities = []
#     for idx, mz_arr, intensity_arr in results:
#         all_mzs.extend(mz_arr.tolist())
#         all_intensities.extend(intensity_arr.tolist())
#
#     # get features from all mzs and intensities
#     feature_mzs = get_msi_features(np.array(all_mzs), np.array(all_intensities), mz_ppm_tol)
#     del all_mzs, all_intensities
#
#     # assign mzs to features
#     args_list = [(idx, *parser.getspectrum(idx), sn_factor, centroided, feature_mzs)
#                  for idx, _ in enumerate(parser.coordinates)]
#
#     # Initialize intensity matrix
#     n_coordinates = len(parser.coordinates)
#     intensity_matrix = np.zeros((len(feature_mzs), n_coordinates))
#
#     # assign mzs to features
#     if n_processes == 1:
#         for args in tqdm(args_list, desc="Feature detection", unit="spectrum"):
#             idx, intensities = assign_spectrum_to_feature(args)
#             intensity_matrix[:, idx] = intensities
#     else:
#         with multiprocessing.Pool(processes=n_processes) as pool:
#             for idx, intensities in tqdm(pool.imap(assign_spectrum_to_feature, args_list),
#                                          total=len(args_list), desc="Feature detection", unit="spectrum"):
#                 intensity_matrix[:, idx] = intensities
#
#     # Get coordinates (x,y only)
#     coordinates = [coord[:2] for coord in parser.coordinates]
#
#     return feature_mzs, intensity_matrix, coordinates


def process_spectra(parser, mz_ppm_tol, sn_factor, centroided, n_processes):
    # first get all spectra and clean them
    args_list = [(idx, *parser.getspectrum(idx), sn_factor, centroided)
                 for idx, _ in enumerate(parser.coordinates)]

    n_processes = n_processes or multiprocessing.cpu_count()

    if n_processes == 1:
        # Non-parallel processing
        results = []
        for args in tqdm(args_list, desc="Denoising spectra", unit="spectrum"):
            results.append(clean_msi_spectrum(args))
    else:
        # Parallel processing with chunks
        chunk_size = min(200, len(parser.coordinates) // n_processes)
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
    feature_mzs = get_msi_features(np.array(all_mzs), np.array(all_intensities), mz_ppm_tol, min_group_size=10,
                                   n_processes=n_processes)
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
        chunk_size = min(200, len(parser.coordinates) // n_processes)
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

    # Get coordinates (x,y only)
    coordinates = [coord[:2] for coord in parser.coordinates]

    return feature_mzs, intensity_matrix, coordinates


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


def save_results(save_dir, mz_values, intensity_matrix, coordinates):
    np.save(os.path.join(save_dir, 'mz_values.npy'), mz_values)
    np.save(os.path.join(save_dir, 'intensity_matrix.npy'), intensity_matrix)
    with open(os.path.join(save_dir, 'coordinates.pkl'), 'wb') as f:
        pickle.dump(coordinates, f)


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
    # process_ms_imaging_data(
    #     "/Users/shipei/Documents/projects/ms1_id/imaging/spotted_stds/2020-12-05_ME_X190_L1_Spotted_20umss_375x450_33at_DAN_Neg.imzML",
    #     "/Users/shipei/Documents/projects/ms1_id/imaging/spotted_stds/2020-12-05_ME_X190_L1_Spotted_20umss_375x450_33at_DAN_Neg.ibd",
    #     n_processes=1)

    feature_mzs = np.load('/Users/shipei/Documents/projects/ms1_id/imaging/feature_mzs.npy')
    print('mz features:', len(feature_mzs))
