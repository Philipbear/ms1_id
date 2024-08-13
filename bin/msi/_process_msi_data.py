import pickle
from collections import defaultdict
import os
import numpy as np
import pyimzml.ImzMLParser as imzml
from numba import njit
from tqdm import tqdm
import multiprocessing


def process_ms_imaging_data(imzml_file, ibd_file, mass_detect_int_tol=None,
                            mz_bin_size=0.005, save=False, save_dir=None,
                            noise_detection='moving_average', n_processes=None):

    validate_inputs(noise_detection)

    parser = imzml.ImzMLParser(imzml_file)

    if save_dir and check_existing_results(save_dir):
        return load_existing_results(save_dir)

    mass_detect_int_tol = auto_detect_intensity_threshold(parser, noise_detection, mass_detect_int_tol)

    mz_intensity_dict, coordinates = process_spectra(parser, noise_detection, mass_detect_int_tol, mz_bin_size,
                                                     n_processes)

    mz_values, intensity_matrix = convert_to_arrays(mz_intensity_dict, coordinates)

    if save and save_dir:
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


def auto_detect_intensity_threshold(parser, noise_detection, mass_detect_int_tol):
    if mass_detect_int_tol is not None:
        return mass_detect_int_tol

    print(f'Auto-denoising MS spectra using {noise_detection} method.')

    if noise_detection == 'percentile':
        return detect_threshold_percentile(parser)
    return None  # For 'moving_average', threshold is determined per spectrum


def detect_threshold_percentile(parser):
    all_intensities = []
    for idx, _ in enumerate(parser.coordinates):
        _, intensity = parser.getspectrum(idx)
        all_intensities.extend(intensity)

    all_intensities = np.array(all_intensities)
    non_zero_intensities = all_intensities[all_intensities > 0.0]
    threshold = max(np.min(non_zero_intensities) * 5, np.percentile(non_zero_intensities, 10))
    print('Auto-detected intensity threshold (percentile method):', threshold)
    return threshold


def process_spectra(parser, noise_detection, mass_detect_int_tol, mz_bin_size, n_processes):
    args_list = [(idx, *parser.getspectrum(idx), noise_detection, mass_detect_int_tol, mz_bin_size)
                 for idx, _ in enumerate(parser.coordinates)]

    n_processes = n_processes or multiprocessing.cpu_count()

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
    idx, mz, intensity, noise_detection, mass_detect_int_tol, mz_bin_size = args

    if noise_detection == 'moving_average':
        baseline = moving_average_baseline(mz, intensity)
        mask = intensity > baseline
    else:
        mask = intensity > mass_detect_int_tol

    # Filter mz and intensity arrays
    filtered_mz = mz[mask]
    filtered_intensity = intensity[mask]

    # Bin m/z values and sum intensities within each bin
    binned_data = defaultdict(float)
    for m, i in zip(filtered_mz, filtered_intensity):
        binned_m = round(m / mz_bin_size) * mz_bin_size
        binned_data[binned_m] += i

    return idx, binned_data


@njit
def moving_average_baseline(mz_array, intensity_array, mz_window=100.0, n_lowest=10, factor=5.0):
    """
    Apply moving average algorithm to a single mass spectrum using an m/z-based window.
    This function is optimized with Numba.

    :param mz_array: numpy array of m/z values
    :param intensity_array: numpy array of intensity values
    :param mz_window: size of the moving window in Da (default: 100.0)
    :param n_lowest: number of lowest points to consider in each window (default: 5)
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
            if len(positive_intensities) >= n_lowest:
                sorted_intensities = np.sort(positive_intensities)
                lowest_n = sorted_intensities[:n_lowest]
                baseline = factor * np.mean(lowest_n)
            else:
                baseline = 0.0
        else:
            baseline = 0.0

        baseline_array.append(baseline)

    return np.array(baseline_array)


def analyze_intensity_distribution(intensity_matrix):
    # Flatten the intensity matrix to get all intensity values
    all_intensities = intensity_matrix.flatten()

    # Remove zero intensities
    non_zero_intensities = all_intensities[all_intensities > 0]

    # Calculate percentiles
    percentiles = [1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99]
    intensity_percentiles = np.percentile(non_zero_intensities, percentiles)

    # Create a dictionary of percentiles and their corresponding intensity values
    percentile_dict = dict(zip(percentiles, intensity_percentiles))

    # Calculate some additional statistics
    mean_intensity = np.mean(non_zero_intensities)
    median_intensity = np.median(non_zero_intensities)
    min_intensity = np.min(non_zero_intensities)
    max_intensity = np.max(non_zero_intensities)

    return {
        "percentiles": percentile_dict,
        "mean": mean_intensity,
        "median": median_intensity,
        "min": min_intensity,
        "max": max_intensity,
        "total_intensities": len(all_intensities),
        "non_zero_intensities": len(non_zero_intensities)
    }


def print_intensity_stats(stats):
    print("Intensity Distribution Analysis:")
    print(f"Total number of intensity values: {stats['total_intensities']}")
    print(f"Number of non-zero intensity values: {stats['non_zero_intensities']}")
    print("\nPercentiles:")
    for percentile, value in stats['percentiles'].items():
        print(f"{percentile}th percentile: {value:.2f}")
    print(f"\nMean intensity: {stats['mean']:.2f}")
    print(f"Median intensity: {stats['median']:.2f}")
    print(f"Minimum non-zero intensity: {stats['min']:.2f}")
    print(f"Maximum intensity: {stats['max']:.2f}")


def create_intensity_histogram(intensity_matrix, bins=1000, percentile_cutoff=95):
    import matplotlib.pyplot as plt
    # Flatten the intensity matrix to get all intensity values
    all_intensities = intensity_matrix.flatten()

    # Remove zero intensities
    non_zero_intensities = all_intensities[all_intensities > 0]

    # Calculate the cutoff value for the x-axis (90th percentile by default)
    x_max = np.percentile(non_zero_intensities, percentile_cutoff)

    # Create the histogram
    plt.figure(figsize=(12, 6))

    # Use range parameter to limit x-axis
    counts, bins, _ = plt.hist(non_zero_intensities, bins=bins, range=(0, x_max), edgecolor='black')

    plt.title(f'Histogram of Non-Zero Intensity Values (First {percentile_cutoff}%)')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    # Add vertical lines for percentiles
    percentiles = [1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
    for percentile, color in zip(percentiles, colors):
        value = np.percentile(non_zero_intensities, percentile)
        if value <= x_max:
            plt.axvline(value, color=color, linestyle='dashed', linewidth=1)
            plt.text(value, plt.gca().get_ylim()[1], f'{percentile}th',
                     rotation=90, va='top', ha='right', color=color)

    # Add text box with statistics
    stats_text = f"Total values: {len(all_intensities)}\n"
    stats_text += f"Non-zero values: {len(non_zero_intensities)}\n"
    stats_text += f"Max shown: {x_max:.2f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    imzml_file = '/Users/shipei/Documents/projects/ms1_id/imaging/mouse_kidney/mouse kidney - root mean square - metaspace.imzML'
    mz_values, intensity_matrix, coordinates, ion_mode = process_ms_imaging_data(
        imzml_file,
        imzml_file.replace('.imzML', '.ibd'),
        mass_detect_int_tol=None
    )

    intensity_stats = analyze_intensity_distribution(intensity_matrix)
    print_intensity_stats(intensity_stats)
    create_intensity_histogram(intensity_matrix, bins=1000, percentile_cutoff=50)
